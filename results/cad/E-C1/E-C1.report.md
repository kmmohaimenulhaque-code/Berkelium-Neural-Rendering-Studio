# E-C1: text -> DesignProposal compiler (Qwen3-8B, NF4 on Kaggle T4)

Scored by the Berkelium core (Manifold backend). Rule: D-033 (pre-registered).

| model | set | n | parse | schema | realise | req | IoU | truncated |
|---|---|---|---|---|---|---|---|---|
| base | test | 451 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0111 |
| base | heldout | 354 | 0.9972 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0367 |
| base | test_gear | 14 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| m2 | test | 451 | 1.0 | 1.0 | 0.9978 | 0.9971 | 0.9357 | 0.0 |
| m2 | heldout | 354 | 0.9887 | 0.6299 | 0.4915 | 0.1494 | 0.0561 | 0.0678 |
| m2 | test_gear | 14 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 |

Paired IoU (m2 - base): {"test": {"n": 451, "iou_diff_mean": 0.9357, "ci95": [0.917, 0.9528]}, "heldout": {"n": 354, "iou_diff_mean": 0.0561, "ci95": [0.0423, 0.0705]}}

## D-033

- test schema >= 0.95: PASS
- test realise >= 0.90: PASS
- test req >= 0.85: PASS
- heldout realise >= 0.60: FAIL
- heldout IoU beats base (paired CI > 0): PASS
- gear req not below base: PASS

**Decision: do not promote**

## Per family (realise / req / IoU)

| family | base | m2 |
|---|---|---|
| bottle | 0.0 / 0.0 / 0.0 | 1.0 / 1.0 / 0.9301 |
| bushing | 0.0 / 0.0 / 0.0 | 1.0 / 1.0 / 0.9643 |
| cantilever_beam | 0.0 / 0.0 / 0.0 | 1.0 / 1.0 / 1.0 |
| chair | 0.0 / 0.0 / 0.0 | 0.5222 / 0.2468 / 0.0255 |
| custom_part | 0.0 / 0.0 / 0.0 | 1.0 / 1.0 / 0.9996 |
| enclosure | 0.0 / 0.0 / 0.0 | 1.0 / 1.0 / 1.0 |
| flange | 0.0 / 0.0 / 0.0 | 1.0 / 1.0 / 0.9777 |
| funnel | 0.0 / 0.0 / 0.0 | 1.0 / 1.0 / 0.984 |
| gear_pair | 0.0 / 0.0 / 0.0 | 1.0 / 1.0 / 1.0 |
| heat_sink | 0.0 / 0.0 / 0.0 | 1.0 / 1.0 / 0.8954 |
| hex_bolt | 0.0 / None / 0.0 | 1.0 / None / 0.9978 |
| hex_nut | 0.0 / 0.0 / 0.0 | 1.0 / 1.0 / 0.9988 |
| hinge_leaf | 0.0 / 0.0 / 0.0 | 0.6333 / 0.0519 / 0.0246 |
| honeycomb_panel | 0.0 / 0.0 / 0.0 | 1.0 / 1.0 / 0.4997 |
| i_beam | 0.0 / 0.0 / 0.0 | 1.0 / 1.0 / 1.0 |
| knob | 0.0 / None / 0.0 | 0.3258 / None / 0.1235 |
| l_bracket | 0.0 / 0.0 / 0.0 | 1.0 / 1.0 / 0.9998 |
| ladder | 0.0 / 0.0 / 0.0 | 1.0 / 1.0 / 0.9993 |
| lantern | 0.0 / None / 0.0 | 1.0 / None / 1.0 |
| mug | 0.0 / 0.0 / 0.0 | 1.0 / 1.0 / 0.9955 |
| pillow_block | 0.0 / None / 0.0 | 1.0 / None / 1.0 |
| pipe_elbow | 0.0 / None / 0.0 | 0.4824 / None / 0.0511 |
| pipe_tee | 0.0 / 0.0 / 0.0 | 1.0 / 1.0 / 0.9995 |
| plate_holes | 0.0 / 0.0 / 0.0 | 1.0 / 1.0 / 0.9952 |
| pressure_vessel | 0.0 / 0.0 / 0.0 | 0.9286 / 0.9 / 0.7809 |
| propeller | 0.0 / 0.0 / 0.0 | 1.0 / 1.0 / 0.9655 |
| pulley | 0.0 / 0.0 / 0.0 | 1.0 / 1.0 / 0.9694 |
| rocket_body | 0.0 / None / 0.0 | 1.0 / None / 0.9901 |
| rocket_nozzle | 0.0 / None / 0.0 | 1.0 / None / 0.2479 |
| spoked_wheel | 0.0 / 0.0 / 0.0 | 1.0 / 1.0 / 0.9995 |
| sprocket | 0.0 / 0.0 / 0.0 | 1.0 / 1.0 / 0.9905 |
| stepped_shaft | 0.0 / 0.0 / 0.0 | 1.0 / 1.0 / 1.0 |
| table | 0.0 / 0.0 / 0.0 | 1.0 / 1.0 / 0.9284 |
| truss | 0.0 / 0.0 / 0.0 | 1.0 / 1.0 / 0.8195 |
| u_channel | 0.0 / 0.0 / 0.0 | 1.0 / 1.0 / 1.0 |
| vase | 0.0 / 0.0 / 0.0 | 1.0 / 1.0 / 0.9994 |
| washer | 0.0 / 0.0 / 0.0 | 1.0 / 1.0 / 1.0 |
