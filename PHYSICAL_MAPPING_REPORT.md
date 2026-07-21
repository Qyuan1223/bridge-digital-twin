# Physical Mapping Model
Samples: 51 (aggressive_001 + conservative_002 + measured test_001 steps 1-6)
Formula: `pose = base(wx, wz, side) + brick_key_offset`; baseline brick `D`.

## Coverage
- `A`: 2 samples
- `A'`: 2 samples
- `B`: 7 samples
- `B'`: 6 samples
- `C`: 4 samples
- `C'`: 4 samples
- `D`: 9 samples
- `E`: 3 samples
- `E'`: 6 samples
- `F`: 5 samples
- `F'`: 3 samples

## Accuracy
- `x`: RMSE 4.76, MAE 3.43, max 16.10
- `y`: RMSE 2.57, MAE 2.09, max 5.46
- `z`: RMSE 2.07, MAE 1.61, max 5.65
- `yaw`: RMSE 2.45, MAE 1.99, max 5.48

## Coefficients

### x
- `intercept`: -200.107132
- `wx`: 7.199623
- `wz`: 0.194374
- `side_R`: 6.502043
- `brick_A`: -21.936378
- `brick_A'`: -13.238892
- `brick_B`: -17.706375
- `brick_B'`: -6.619005
- `brick_C`: -19.008249
- `brick_C'`: -37.218992
- `brick_E`: -14.729027
- `brick_E'`: 1.053153
- `brick_F`: -13.167235
- `brick_F'`: -10.864115

### y
- `intercept`: -203.088894
- `wx`: -0.043181
- `wz`: -0.676320
- `side_R`: 0.235569
- `brick_A`: 1.067539
- `brick_A'`: 0.665380
- `brick_B`: 0.054836
- `brick_B'`: -0.360441
- `brick_C`: 1.741643
- `brick_C'`: -1.642579
- `brick_E`: -1.291962
- `brick_E'`: -1.558979
- `brick_F`: 0.881590
- `brick_F'`: -0.264911

### z
- `intercept`: 102.831448
- `wx`: -0.124176
- `wz`: 7.630503
- `side_R`: 3.232967
- `brick_A`: 5.620533
- `brick_A'`: -0.603541
- `brick_B`: -0.750239
- `brick_B'`: 1.423885
- `brick_C`: 1.164171
- `brick_C'`: 0.389611
- `brick_E`: 0.228321
- `brick_E'`: 0.850260
- `brick_F`: 4.357944
- `brick_F'`: 3.041310

### yaw
- `intercept`: 97.139426
- `wx`: 1.438801
- `wz`: 0.117002
- `side_R`: 5.076226
- `brick_A`: -6.442845
- `brick_A'`: -4.391792
- `brick_B`: -4.329968
- `brick_B'`: -3.698316
- `brick_C`: -4.050894
- `brick_C'`: -7.857166
- `brick_E`: -3.690507
- `brick_E'`: -0.604786
- `brick_F`: -2.843668
- `brick_F'`: -0.227521

## Notes
- `A'` now has measured samples from test_001, so it no longer falls back to `A`.
- `E` coverage improved from 1 sample to 3 samples.
- Test_001 steps 7-11 are still pending and were not used in this fit.
- Direct measured JSON remains safer than predicted poses for full-bridge demos.
