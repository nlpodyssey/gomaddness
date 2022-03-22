// Copyright 2022, NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gomaddness

// LookupTable holds a table of pre-computed dot products, quantized to 8 bits,
// and the parameters needed for de-quantization during the product
// quantization's aggregation step.
type LookupTable[F Float] struct {
	Bias  F
	Scale F
	// Matrix, represented in row-major order, with row index = subspace,
	// and column index = prototype.
	Data []uint8
}
