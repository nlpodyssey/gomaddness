// Copyright 2022, NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gomaddness

// A Bucket is a set of Vectors, supporting the learning process of
// MADDNESS hash function parameters.
//
// The tree Level, and the NodeIndex within that level, are included
// as a convenient for debugging or logging purposes.
type Bucket[F Float] struct {
	Level     int
	NodeIndex int
	Vectors   Vectors[F]
}
