/*
Copyright 2021 BlackRock, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

use std::f64;

pub fn approx_ge(x: f64, y: f64) -> bool {
    x > y || relative_eq!(x, y, epsilon = f64::EPSILON)
}

pub fn approx_le(x: f64, y: f64) -> bool {
    approx_ge(y, x)
}

pub fn gt_ep(x: f64, y: f64) -> bool {
    x - y > f64::EPSILON
}

pub fn lt_ep(x: f64, y: f64) -> bool {
    gt_ep(y, x)
}

pub fn clamp(x: f64, min: f64, max: f64) -> f64 {
    if lt_ep(x, min) {
        min
    } else if gt_ep(x, max) {
        max
    } else {
        x
    }
}
