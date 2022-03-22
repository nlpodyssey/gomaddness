// Copyright 2022, NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gomaddness

import (
	"log"
	"math"
	"runtime"
)

// Maddness is the primary structure that holds parameters and implements
// methods of the whole MADDNESS algorithm.
type Maddness[F Float] struct {
	NumSubspaces  int
	VectorSize    int
	SubVectorSize int
	Hashes        []*Hash[F]
	LookupTables  []*LookupTable[F]
}

// TrainMaddness runs the learning process for MADDNESS product quantization and
// hash functions parameters, returning a new trained Maddness object.
func TrainMaddness[F Float](dataExamples, queryVectors Vectors[F], numSubspaces int) *Maddness[F] {
	log.Printf("maddness: training starts.")

	if len(dataExamples) == 0 {
		panic("maddness: invalid empty dataExamples")
	}
	if len(queryVectors) == 0 {
		panic("maddness: invalid empty queryVectors")
	}

	vecSize := len(dataExamples[0])
	if vecSize == 0 {
		panic("maddness: invalid zero vector size")
	}

	if numSubspaces < 0 || numSubspaces > vecSize || vecSize%numSubspaces != 0 {
		panic("maddness: invalid numSubspaces (it must be a positive factor of vectors' length)")
	}

	m := &Maddness[F]{
		NumSubspaces:  numSubspaces,
		VectorSize:    vecSize,
		SubVectorSize: vecSize / numSubspaces,
	}

	m.trainAllHashes(dataExamples)
	m.makeLookupTables(queryVectors)

	return m
}

// Quantize splits the given vector into subspaces and returns a slice
// of hash indices, one for each subspace.
func (m *Maddness[F]) Quantize(v Vector[F]) []uint8 {
	hashes := m.Hashes
	subVectorSize := m.SubVectorSize

	q := make([]uint8, len(hashes))
	for i, hash := range hashes {
		offset := i * subVectorSize
		subVector := v[offset : offset+subVectorSize]
		q[i] = hash.Hash(subVector)
	}
	return q
}

// LookupTableIndices transforms a list of hash indices, as returned from
// Quantize, into a corresponding list of lookup-table indices,
// for accessing LookupTable.Data.
func (m *Maddness[F]) LookupTableIndices(q []uint8) []uint16 {
	lutCols := len(m.Hashes[0].Prototypes)
	indices := make([]uint16, len(q))
	for subspaceIndex, protoIndex := range q {
		indices[subspaceIndex] = uint16(subspaceIndex*lutCols) + uint16(protoIndex)
	}
	return indices
}

// DotProduct computes the approximated dot product between a data vector,
// identified by the lookup-table indices obtained from the vector's
// quantization, and the query vector represented by queryVectorIndex.
func (m *Maddness[F]) DotProduct(lutIndices []uint16, queryVectorIndex int) F {
	lut := m.LookupTables[queryVectorIndex]
	lutData := lut.Data

	var sum uint16
	for _, lutIndex := range lutIndices {
		ldi := lutData[lutIndex]
		sum += uint16(ldi)
	}

	return F(sum)/lut.Scale + lut.Bias
}

// Reconstruct builds a vector from a list of hash indices, reconstructed
// using the learned prototypes for each subspace.
func (m *Maddness[F]) Reconstruct(q []uint8) Vector[F] {
	v := make(Vector[F], 0, m.VectorSize)
	for i, hash := range m.Hashes {
		v = append(v, hash.Prototypes[q[i]]...)
	}
	return v
}

func (m *Maddness[F]) trainAllHashes(examples Vectors[F]) {
	log.Printf("maddness: training %d subspaces with %d examples...", m.NumSubspaces, len(examples))

	// Use a channel to limit concurrency.
	concurrency := runtime.NumCPU()
	ch := make(chan struct{}, concurrency)

	m.Hashes = make([]*Hash[F], m.NumSubspaces)
	for i := range m.Hashes {
		ch <- struct{}{} // reserve one working slot, or wait for a free one
		go func(subIndex int) {
			m.trainSubspaceHash(subIndex, examples)
			<-ch // free the slot
		}(i)
	}

	// Wait until all work is done
	for i := 0; i < concurrency; i++ {
		ch <- struct{}{}
	}
	close(ch)
	log.Print("maddness: subspaces training completed.")
}

func (m *Maddness[F]) trainSubspaceHash(subIndex int, allExamples Vectors[F]) {
	log.Printf("maddness: training subspace %d of %d...", subIndex+1, m.NumSubspaces)

	subExamples := m.subspaceExamples(subIndex, allExamples)
	m.Hashes[subIndex] = TrainHash(subExamples)
}

func (m *Maddness[F]) makeLookupTables(queryVectors Vectors[F]) {
	log.Printf("maddness: creating lookup tables with %d query vectors...", len(queryVectors))

	m.LookupTables = make([]*LookupTable[F], len(queryVectors))
	for i, qv := range queryVectors {
		m.LookupTables[i] = m.makeLookupTable(qv)
	}

	log.Print("maddness: lookup tables created.")
}

func (m *Maddness[F]) makeLookupTable(queryVector Vector[F]) *LookupTable[F] {
	floatData, min, scale := m.precomputeDotProducts(queryVector)

	rows := len(floatData)
	cols := len(floatData[0])

	data := make([]uint8, 0, rows*cols)
	for _, fdRow := range floatData {
		for _, v := range fdRow {
			data = append(data, uint8((v-min)*scale))
		}
	}

	return &LookupTable[F]{
		Bias:  min * F(m.NumSubspaces),
		Scale: scale,
		Data:  data,
	}
}

func (m *Maddness[F]) precomputeDotProducts(vec Vector[F]) (data Vectors[F], min, scale F) {
	data = make(Vectors[F], m.NumSubspaces)
	min = F(math.Inf(1))
	max := F(math.Inf(-1))

	for i := range data {
		subOffset := i * m.SubVectorSize
		subVec := vec[subOffset : subOffset+m.SubVectorSize]

		protos := m.Hashes[i].Prototypes
		dataRow := make(Vector[F], len(protos))
		for j, proto := range protos {
			v := subVec.DotProduct(proto)
			dataRow[j] = v
			if v < min {
				min = v
			}
			if v > max {
				max = v
			}
		}
		data[i] = dataRow
	}
	scale = math.MaxUint8 / (max - min)
	return
}

func (m *Maddness[F]) subspaceExamples(subIndex int, allExamples Vectors[F]) Vectors[F] {
	offset := subIndex * m.SubVectorSize
	end := offset + m.SubVectorSize

	subExamples := make(Vectors[F], len(allExamples))
	for i, ex := range allExamples {
		subExamples[i] = ex[offset:end]
	}

	return subExamples
}
