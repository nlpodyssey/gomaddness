// Copyright 2022, NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gomaddness

// Buckets is a set of Bucket items. Usually, they all belong to the same
// level of the hashing tree, built during the learning process.
type Buckets[F Float] []*Bucket[F]

// HeuristicSelectIndices selects a set of indices to be evaluated during
// the construction of a hashing-tree level.
//
// It returns a maximum of top-four indices that contribute the most loss,
// summed across all Buckets.
func (bs Buckets[_]) HeuristicSelectIndices() []int {
	sumOfVariance := bs[0].Vectors.ColumnWiseVariance()
	for _, b := range bs[1:] {
		sumOfVariance.Add(b.Vectors.ColumnWiseVariance())
	}
	return NewArgMaxHeap(sumOfVariance).FirstArgsMax(4)
}

// Prototypes creates the prototype Vectors.
//
// For each Bucket, a prototype Vector is created by computing the mean
// of its sub-vectors.
func (bs Buckets[F]) Prototypes() Vectors[F] {
	protos := make(Vectors[F], len(bs))
	for i, bucket := range bs {
		protos[i] = bucket.Vectors.Mean()
	}
	return protos
}
