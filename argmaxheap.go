// Copyright 2022, NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gomaddness

import "container/heap"

// ArgMaxHeap is a max-heap working on the indices of a Vector, without
// modifying the Vector itself.
type ArgMaxHeap[F Float] struct {
	vector  Vector[F]
	indices []int
}

// NewArgMaxHeap creates and initializes a new ArgMaxHeap object.
func NewArgMaxHeap[F Float](v Vector[F]) *ArgMaxHeap[F] {
	indices := make([]int, len(v))
	for i := range indices {
		indices[i] = i
	}
	h := &ArgMaxHeap[F]{
		vector:  v,
		indices: indices,
	}
	heap.Init(h)
	return h
}

// FirstArgsMax returns the first n arguments of the maxima (arg max) of v.
//
// If n is greater than Len, only Len indices are returned.
func (h *ArgMaxHeap[_]) FirstArgsMax(n int) []int {
	if n > len(h.indices) {
		n = len(h.indices)
	}
	indices := make([]int, n)
	for i := range indices {
		indices[i] = heap.Pop(h).(int)
	}
	return indices
}

// Len is the number of indices in the collection.
func (h *ArgMaxHeap[_]) Len() int {
	return len(h.indices)
}

// Less reports whether the index at i must sort before the index of j.
// It returns true only if the vector's value at the i-th index is grater
// than the value at the j-th index.
func (h *ArgMaxHeap[_]) Less(i, j int) bool {
	return h.vector[h.indices[i]] > h.vector[h.indices[j]]
}

// Swap swaps the i-th and j-th indices.
func (h *ArgMaxHeap[_]) Swap(i, j int) {
	h.indices[i], h.indices[j] = h.indices[j], h.indices[i]
}

// Push always panics: in order to simplify the implementation, it is not
// allowed to add new elements.
func (h *ArgMaxHeap[_]) Push(any) {
	panic("maddness: unexpected call to ArgMaxHeap.Push")
}

// Pop removes and returns the index of the maximum element
// (according to Less) from the heap.
func (h *ArgMaxHeap[_]) Pop() any {
	lastIndex := len(h.indices) - 1
	x := h.indices[lastIndex]
	h.indices = h.indices[:lastIndex]
	return x
}
