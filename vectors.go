// Copyright 2022, NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gomaddness

import (
	"math"
	"sort"
)

// Vectors is a collection of Vector objects.
//
// All Vector objects MUST be of the same size.
//
// Given a collection of N vectors, each of size M, this can also be seen as a
// matrix with dimensions N×M (N rows, M columns).
type Vectors[F Float] []Vector[F]

// ColumnWiseVariance returns a new Vector containing the column-wise variance
// computed across all vectors.
func (vs Vectors[F]) ColumnWiseVariance() Vector[F] {
	vecSize := len(vs[0])
	sum := make(Vector[F], vecSize)
	sumOfSquares := make(Vector[F], vecSize)

	for _, v := range vs {
		sum.Add(v)
		sumOfSquares.AddSquares(v)
	}

	n := F(len(vs))
	mean := sum.DivScalar(n)
	squareMean := mean.Square()
	return sumOfSquares.DivScalar(n).Sub(squareMean)
}

// SplitByThreshold splits the collection of vectors by a threshold value,
// compared with the elements at the given index.
//
// Two new collections are returned:
//   - lt: vectors whose value at "index" is lower than "threshold"
//   - gte: vectors whose value at "index" is greater than or equal to "threshold"
func (vs Vectors[F]) SplitByThreshold(index int, threshold F) (lt, gte Vectors[F]) {
	lt = make(Vectors[F], 0)
	gte = make(Vectors[F], 0)
	for _, v := range vs {
		if v[index] < threshold {
			lt = append(lt, v)
			continue
		}
		gte = append(gte, v)
	}
	return
}

// Copy returns a shallow copy of the collection.
func (vs Vectors[F]) Copy() Vectors[F] {
	c := make(Vectors[F], len(vs))
	copy(c, vs)
	return c
}

// SortByColumn sorts vs in place, according to the value at each vector's
// index (column), in ascending order.
// It returns vs.
func (vs Vectors[F]) SortByColumn(index int) Vectors[F] {
	sort.SliceStable(vs, func(i, j int) bool {
		return vs[i][index] < vs[j][index]
	})
	return vs
}

// Reverse reverses vs in place.
// It returns vs.
func (vs Vectors[F]) Reverse() Vectors[F] {
	for l, r := 0, len(vs)-1; l < r; l, r = l+1, r-1 {
		vs[l], vs[r] = vs[r], vs[l]
	}
	return vs
}

// CumulativeSSE returns a new Vector, computing the error sum of squares
// (SSE) over the set of Vectors.
func (vs Vectors[F]) CumulativeSSE() Vector[F] {
	n := len(vs)
	d := len(vs[0])

	out := make(Vector[F], n) // all values are zero by default
	sum := make(Vector[F], d)
	sumOfSquares := make(Vector[F], d)

	_ = sum[len(vs[0])-1]
	for j, x := range vs[0] {
		sum[j] = x
		sumOfSquares[j] = x * x
	}

	for i := 1; i < n; i++ {
		_ = sum[len(vs[i])-1]
		for j, x := range vs[i] {
			sum[j] += x
			sumOfSquares[j] += x * x
			out[i] += sumOfSquares[j] - (sum[j] * sum[j] / F(i+1))
		}
	}

	return out
}

// OptimalSplitThreshold finds the optimal split threshold within the vectors,
// computed over values at the given split-index column.
func (vs Vectors[F]) OptimalSplitThreshold(splitIndex int) (threshold, loss F) {
	sorted := vs.Copy().SortByColumn(splitIndex)
	if len(sorted) == 1 {
		// TODO: check this corner case
		sorted = append(sorted, sorted[0])
	}

	losses := sorted.CumulativeSSE() // head
	tail := sorted.Copy().Reverse().CumulativeSSE().Reverse()

	losses[:len(losses)-1].Add(tail[1:])

	n := losses.ArgMin()
	if n < len(sorted)-1 {
		threshold = (sorted[n][splitIndex] + sorted[n+1][splitIndex]) / 2
	} else {
		// TODO: check this corner case
		threshold = F(math.Nextafter32(float32(sorted[n][splitIndex]), float32(math.Inf(+1))))
	}
	return threshold, losses[n]
}

// Mean calculates and returns the mean vector.
func (vs Vectors[F]) Mean() Vector[F] {
	sum := vs[0].Copy()
	for _, v := range vs[1:] {
		sum.Add(v)
	}
	return sum.DivScalar(F(len(vs)))
}
