# SEPA-Eval Report — 2026-05-24

## 1. Capability Frontier
| Model | Benchmark | Task | SR | Clean SR | Trials |
|-------|-----------|------|----|----------|--------|
| NeuroVLA-v1.2 | libero_goal | lg_arrange_fruit_bowl | 0.500 | 0.485 | 2 |
| NeuroVLA-v1.2 | libero_goal | lg_bowl_in_microwave | 0.500 | 0.485 | 2 |
| NeuroVLA-v1.2 | libero_goal | lg_butter_in_fridge | 0.500 | 0.485 | 2 |
| NeuroVLA-v1.2 | libero_goal | lg_close_cabinet | 0.500 | 0.485 | 2 |
| NeuroVLA-v1.2 | libero_goal | lg_cup_in_rack | 1.000 | 0.970 | 2 |
| NeuroVLA-v1.2 | libero_goal | lg_kettle_on_plate | 0.500 | 0.485 | 2 |
| NeuroVLA-v1.2 | libero_goal | lg_lift_book_stack | 0.000 | 0.000 | 2 |
| NeuroVLA-v1.2 | libero_goal | lg_open_oven_door | 0.000 | 0.000 | 2 |
| NeuroVLA-v1.2 | libero_goal | lg_plate_on_stove | 1.000 | 0.970 | 2 |
| NeuroVLA-v1.2 | libero_goal | lg_pour_kettle | 0.000 | 0.000 | 2 |
| QwenOFT-v2.1 | libero_goal | lg_arrange_fruit_bowl | 0.000 | 0.000 | 2 |
| QwenOFT-v2.1 | libero_goal | lg_bowl_in_microwave | 0.500 | 0.485 | 2 |
| QwenOFT-v2.1 | libero_goal | lg_butter_in_fridge | 0.500 | 0.485 | 2 |
| QwenOFT-v2.1 | libero_goal | lg_close_cabinet | 1.000 | 0.970 | 2 |
| QwenOFT-v2.1 | libero_goal | lg_cup_in_rack | 1.000 | 0.970 | 2 |
| QwenOFT-v2.1 | libero_goal | lg_kettle_on_plate | 0.000 | 0.000 | 2 |
| QwenOFT-v2.1 | libero_goal | lg_lift_book_stack | 0.500 | 0.485 | 2 |
| QwenOFT-v2.1 | libero_goal | lg_open_oven_door | 0.500 | 0.485 | 2 |
| QwenOFT-v2.1 | libero_goal | lg_plate_on_stove | 1.000 | 0.970 | 2 |
| QwenOFT-v2.1 | libero_goal | lg_pour_kettle | 0.500 | 0.485 | 2 |
| NeuroVLA-v1.2 | libero_spatial | ls_move_block_corner | 1.000 | 0.970 | 3 |
| NeuroVLA-v1.2 | libero_spatial | ls_move_plate_behind | 1.000 | 0.970 | 3 |
| NeuroVLA-v1.2 | libero_spatial | ls_pick_cup_top_shelf | 1.000 | 0.970 | 3 |
| NeuroVLA-v1.2 | libero_spatial | ls_pick_mug_right_shelf | 0.667 | 0.647 | 3 |
| NeuroVLA-v1.2 | libero_spatial | ls_pick_red_cup_left | 1.000 | 0.970 | 3 |
| NeuroVLA-v1.2 | libero_spatial | ls_place_bowl_front | 1.000 | 0.970 | 3 |
| NeuroVLA-v1.2 | libero_spatial | ls_place_knife_right | 1.000 | 0.970 | 3 |
| NeuroVLA-v1.2 | libero_spatial | ls_put_butter_right | 1.000 | 0.970 | 3 |
| NeuroVLA-v1.2 | libero_spatial | ls_put_can_rightmost | 1.000 | 0.970 | 3 |
| NeuroVLA-v1.2 | libero_spatial | ls_slide_book_left | 0.667 | 0.647 | 3 |
| QwenOFT-v2.1 | libero_spatial | ls_move_block_corner | 1.000 | 0.970 | 3 |
| QwenOFT-v2.1 | libero_spatial | ls_move_plate_behind | 1.000 | 0.970 | 3 |
| QwenOFT-v2.1 | libero_spatial | ls_pick_cup_top_shelf | 1.000 | 0.970 | 3 |
| QwenOFT-v2.1 | libero_spatial | ls_pick_mug_right_shelf | 0.667 | 0.647 | 3 |
| QwenOFT-v2.1 | libero_spatial | ls_pick_red_cup_left | 1.000 | 0.970 | 3 |
| QwenOFT-v2.1 | libero_spatial | ls_place_bowl_front | 1.000 | 0.970 | 3 |
| QwenOFT-v2.1 | libero_spatial | ls_place_knife_right | 0.667 | 0.647 | 3 |
| QwenOFT-v2.1 | libero_spatial | ls_put_butter_right | 0.667 | 0.647 | 3 |
| QwenOFT-v2.1 | libero_spatial | ls_put_can_rightmost | 1.000 | 0.970 | 3 |
| QwenOFT-v2.1 | libero_spatial | ls_slide_book_left | 1.000 | 0.970 | 3 |

## 2. Saturation Map
| Task | Benchmark | Status | Disc. Power | Saturated |
|------|-----------|--------|-------------|-----------|
| ls_pick_red_cup_left | libero_spatial | seed | 0.000 | Yes |
| ls_place_bowl_front | libero_spatial | seed | 0.000 | Yes |
| ls_move_plate_behind | libero_spatial | seed | 0.000 | Yes |
| ls_put_can_rightmost | libero_spatial | seed | 0.000 | Yes |
| ls_pick_cup_top_shelf | libero_spatial | seed | 0.000 | Yes |
| ls_move_block_corner | libero_spatial | seed | 0.000 | Yes |
| lg_plate_on_stove | libero_goal | seed | 0.000 | Yes |
| lg_cup_in_rack | libero_goal | seed | 0.000 | Yes |
| ls_pick_mug_right_shelf | libero_spatial | seed | 0.000 | No |
| lg_bowl_in_microwave | libero_goal | seed | 0.000 | No |
| lg_butter_in_fridge | libero_goal | seed | 0.000 | No |
| ls_put_butter_right | libero_spatial | seed | 0.167 | No |
| ls_slide_book_left | libero_spatial | seed | 0.167 | No |
| ls_place_knife_right | libero_spatial | seed | 0.167 | No |
| lg_open_oven_door | libero_goal | seed | 0.250 | No |
| lg_kettle_on_plate | libero_goal | seed | 0.250 | No |
| lg_close_cabinet | libero_goal | seed | 0.250 | No |
| lg_lift_book_stack | libero_goal | seed | 0.250 | No |
| lg_pour_kettle | libero_goal | seed | 0.250 | No |
| lg_arrange_fruit_bowl | libero_goal | seed | 0.250 | No |
| 7af07b11-f547-4c9f-be6b-30d5460ef602 | libero_goal | promoted | 0.350 | No |
| ae88699f-f204-478c-bf13-bf078c4e55ec | libero_goal | promoted | 0.381 | No |
| e16ceb30-e55b-4c4b-a5a6-f8d129764d5a | libero_goal | candidate | 0.491 | No |

## 3. Failure Taxonomy
**NeuroVLA-v1.2**
  - timeout: 5
  - out_of_reach: 3
  - grasp: 2
  - contact_dynamics: 2
  - recovery: 1
**QwenOFT-v2.1**
  - contact_dynamics: 4
  - timeout: 3
  - grasp: 3
  - pose_estimation: 1
  - out_of_reach: 1

## 4. Evolved Task Summary
- Seeds analyzed: 20
- Candidates generated: 23
- Tasks promoted: 2

## 5. Cross-Model Failure Heatmap
| Task | NeuroVLA-v1.2 | QwenOFT-v2.1 |
|------|------|------|
| `lg_arrange_fruit_bowl` | ⚠️ 0.50 | ❌ 0.00 |
| `lg_bowl_in_microwave` | ⚠️ 0.50 | ⚠️ 0.50 |
| `lg_butter_in_fridge` | ⚠️ 0.50 | ⚠️ 0.50 |
| `lg_close_cabinet` | ⚠️ 0.50 | ✅ 1.00 |
| `lg_cup_in_rack` | ✅ 1.00 | ✅ 1.00 |
| `lg_kettle_on_plate` | ⚠️ 0.50 | ❌ 0.00 |
| `lg_lift_book_stack` | ❌ 0.00 | ⚠️ 0.50 |
| `lg_open_oven_door` | ❌ 0.00 | ⚠️ 0.50 |
| `lg_plate_on_stove` | ✅ 1.00 | ✅ 1.00 |
| `lg_pour_kettle` | ❌ 0.00 | ⚠️ 0.50 |
| `ls_move_block_corner` | ✅ 1.00 | ✅ 1.00 |
| `ls_move_plate_behind` | ✅ 1.00 | ✅ 1.00 |
| `ls_pick_cup_top_shelf` | ✅ 1.00 | ✅ 1.00 |
| `ls_pick_mug_right_shelf` | ⚠️ 0.67 | ⚠️ 0.67 |
| `ls_pick_red_cup_left` | ✅ 1.00 | ✅ 1.00 |
| `ls_place_bowl_front` | ✅ 1.00 | ✅ 1.00 |
| `ls_place_knife_right` | ✅ 1.00 | ⚠️ 0.67 |
| `ls_put_butter_right` | ✅ 1.00 | ⚠️ 0.67 |
| `ls_put_can_rightmost` | ✅ 1.00 | ✅ 1.00 |
| `ls_slide_book_left` | ⚠️ 0.67 | ✅ 1.00 |

**NeuroVLA-v1.2 only** (model-specific weakness): `lg_lift_book_stack`, `lg_open_oven_door`, `lg_pour_kettle`
**QwenOFT-v2.1 only** (model-specific weakness): `lg_arrange_fruit_bowl`, `lg_kettle_on_plate`

---
*Generated by SEPA-Eval on 2026-05-24T17:40:41Z*
