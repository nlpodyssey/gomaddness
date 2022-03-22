// Copyright 2022, NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gomaddness

// Float is a constraint that permits any floating-point type.
type Float interface {
	~float32 | ~float64
}
