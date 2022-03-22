// Copyright 2022, NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gomaddness

import "math"

// Hash is the data structure for MADDNESS hash function.
// It holds the learned balanced binary regression tree and the prototype
// vectors.
type Hash[F Float] struct {
	TreeLevels []*HashingTreeLevel[F]
	Prototypes Vectors[F]
}

// HashingTreeLevel is one level of the binary tree from a Hash.
type HashingTreeLevel[F Float] struct {
	SplitIndex      int
	SplitThresholds Vector[F]
}

// TrainHash runs the learning process for MADDNESS hash function parameters,
// and return a new trained Hash.
func TrainHash[F Float](examples Vectors[F]) *Hash[F] {
	buckets := Buckets[F]{
		&Bucket[F]{
			Level:     -1,
			NodeIndex: 0,
			Vectors:   examples,
		},
	}

	levels := make([]*HashingTreeLevel[F], 4)
	for i := range levels {
		buckets, levels[i] = nextHashingTreeLevel(buckets)
	}

	return &Hash[F]{
		TreeLevels: levels,
		Prototypes: buckets.Prototypes(),
	}
}

// Hash maps the given vector to an index, applying MADDNESS hash function.
func (h *Hash[F]) Hash(v Vector[F]) uint8 {
	var i uint8 = 1
	for _, level := range h.TreeLevels {
		threshold := level.SplitThresholds[i-1]
		i = 2 * i
		if v[level.SplitIndex] < threshold {
			i--
			continue
		}
	}
	return i - 1
}

func nextHashingTreeLevel[F Float](buckets Buckets[F]) (Buckets[F], *HashingTreeLevel[F]) {
	indices := buckets.HeuristicSelectIndices()

	bestLoss := F(math.Inf(+1))
	bestSplitIndex := -1
	var bestSplitThresholds Vector[F]

	for _, splitIndex := range indices {
		var loss F
		splitThresholds := make(Vector[F], len(buckets))
		for j, bucket := range buckets {
			t, l := bucket.Vectors.OptimalSplitThreshold(splitIndex)
			splitThresholds[j] = t
			loss += l
		}
		if loss < bestLoss {
			bestLoss = loss
			bestSplitIndex = splitIndex
			bestSplitThresholds = splitThresholds
		}
	}

	newBuckets := make(Buckets[F], 0, len(buckets)*2)
	for j, bucket := range buckets {
		lt, gte := bucket.Vectors.SplitByThreshold(bestSplitIndex, bestSplitThresholds[j])

		// TODO: check corner cases when lt or gte are empty
		if len(lt) == 0 {
			v := gte.Copy().SortByColumn(bestSplitIndex)[0].Copy()
			v[bestSplitIndex] = F(math.Nextafter32(float32(v[bestSplitIndex]), float32(math.Inf(-1))))
			lt = Vectors[F]{v}
		}
		if len(gte) == 0 {
			v := lt.Copy().SortByColumn(bestSplitIndex)[len(lt)-1].Copy()
			v[bestSplitIndex] = F(math.Nextafter32(float32(v[bestSplitIndex]), float32(math.Inf(+1))))
			gte = Vectors[F]{v}
		}

		newBuckets = append(newBuckets, &Bucket[F]{
			Level:     bucket.Level + 1,
			NodeIndex: j * 2,
			Vectors:   lt,
		})
		newBuckets = append(newBuckets, &Bucket[F]{
			Level:     bucket.Level + 1,
			NodeIndex: j*2 + 1,
			Vectors:   gte,
		})
	}

	nextLevel := &HashingTreeLevel[F]{
		SplitIndex:      bestSplitIndex,
		SplitThresholds: bestSplitThresholds,
	}
	return newBuckets, nextLevel
}
