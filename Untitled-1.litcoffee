# palak@audit-4-nodes-0-0:~/audit$ python3 step_5_audit.py --config config.yaml

# Step 5: Black-Box Fairness Auditing

Config : config.yaml
Experiment : lenet_alpha0.5_5nodes_seed42
NFS root : /mnt/nfs/home/palak/data/models/vllm/
Audit modes : full_local, budgeted, global
Budget sizes : [100, 500, 1000, 2000, 5000]
Repeats/budget : 10
Log file : /mnt/nfs/home/palak/data/models/vllm/experiments/lenet_alpha0.5_5nodes_seed42/logs/step5.log

GPUs available : 4
GPU 0: NVIDIA A100-SXM4-80GB
GPU 1: NVIDIA A100-SXM4-80GB
GPU 2: NVIDIA A100-SXM4-80GB
GPU 3: NVIDIA A100-SXM4-80GB

Total audits : 1,025
Full local : 20
Budgeted : 1,000
Global : 5

Loading CelebA...
HTTP Request: HEAD https://huggingface.co/datasets/flwrlabs/celeba/resolve/main/README.md "HTTP/1.1 307 Temporary Redirect"
HTTP Request: HEAD https://huggingface.co/api/resolve-cache/datasets/flwrlabs/celeba/2d738f56e0e7f925ea36ae7c808ea925264aacec/README.md "HTTP/1.1 200 OK"
HTTP Request: HEAD https://huggingface.co/datasets/flwrlabs/celeba/resolve/2d738f56e0e7f925ea36ae7c808ea925264aacec/celeba.py "HTTP/1.1 404 Not Found"
HTTP Request: HEAD https://s3.amazonaws.com/datasets.huggingface.co/datasets/datasets/flwrlabs/celeba/flwrlabs/celeba.py "HTTP/1.1 404 Not Found"
HTTP Request: GET https://huggingface.co/api/datasets/flwrlabs/celeba/revision/2d738f56e0e7f925ea36ae7c808ea925264aacec "HTTP/1.1 200 OK"
HTTP Request: HEAD https://huggingface.co/datasets/flwrlabs/celeba/resolve/2d738f56e0e7f925ea36ae7c808ea925264aacec/.huggingface.yaml "HTTP/1.1 404 Not Found"
HTTP Request: GET https://datasets-server.huggingface.co/info?dataset=flwrlabs/celeba "HTTP/1.1 200 OK"
HTTP Request: GET https://huggingface.co/api/datasets/flwrlabs/celeba/tree/2d738f56e0e7f925ea36ae7c808ea925264aacec/img_align%2Bidentity%2Battr?recursive=true&expand=false "HTTP/1.1 200 OK"
HTTP Request: GET https://huggingface.co/api/datasets/flwrlabs/celeba/tree/2d738f56e0e7f925ea36ae7c808ea925264aacec?recursive=false&expand=false "HTTP/1.1 200 OK"
Resolving data files: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 19/19 [00:00<00:00, 227690.79it/s]
HTTP Request: HEAD https://huggingface.co/datasets/flwrlabs/celeba/resolve/2d738f56e0e7f925ea36ae7c808ea925264aacec/dataset_infos.json "HTTP/1.1 404 Not Found"
Resolving data files: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 19/19 [00:00<00:00, 240760.65it/s]
Loading dataset shards: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 19/19 [00:00<00:00, 378.14it/s]
✓ 162,770 samples loaded
✓ Partition loaded from /mnt/nfs/home/palak/data/models/vllm/experiments/lenet_alpha0.5_5nodes_seed42/partitions/partition_alpha0.5_seed42.json

True DP gaps from Step 4:
Node Data Model Val Model Full

---

Node 1 0.1448 0.1632 0.1362
Node 2 0.1405 0.1285 0.1350
Node 3 0.1329 0.1628 0.1140
Node 4 0.1360 0.2301 0.1044
Node 5 0.1504 0.1616 0.1487

──────────────────────────────────────────────────────────────────────
Launching 5 audit workers (max 4 parallel)
──────────────────────────────────────────────────────────────────────

▶ Launched Target 1 on GPU 0 (pid=31839)
/home/palak/audit/step_5_audit.py:270: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
model.load_state_dict(torch.load(ckpt_path, map_location=device))
[Target 1 | GPU 0] Model loaded from /mnt/nfs/home/palak/data/models/vllm/experiments/lenet_alpha0.5_5nodes_seed42/checkpoints/node_1_best.pt
▶ Launched Target 2 on GPU 1 (pid=31983)
/home/palak/audit/step_5_audit.py:270: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
model.load_state_dict(torch.load(ckpt_path, map_location=device))
[Target 2 | GPU 1] Model loaded from /mnt/nfs/home/palak/data/models/vllm/experiments/lenet_alpha0.5_5nodes_seed42/checkpoints/node_2_best.pt
▶ Launched Target 3 on GPU 2 (pid=32138)
/home/palak/audit/step_5_audit.py:270: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
model.load_state_dict(torch.load(ckpt_path, map_location=device))
[Target 3 | GPU 2] Model loaded from /mnt/nfs/home/palak/data/models/vllm/experiments/lenet_alpha0.5_5nodes_seed42/checkpoints/node_3_best.pt
▶ Launched Target 4 on GPU 3 (pid=32293)
/home/palak/audit/step_5_audit.py:270: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
model.load_state_dict(torch.load(ckpt_path, map_location=device))
[Target 4 | GPU 3] Model loaded from /mnt/nfs/home/palak/data/models/vllm/experiments/lenet_alpha0.5_5nodes_seed42/checkpoints/node_4_best.pt
[Target 2 | GPU 1] Full local | Auditor 1 → est_dp=0.1393 abs_err=0.0108
[Target 1 | GPU 0] Full local | Auditor 2 → est_dp=0.1312 abs_err=0.0320
[Target 3 | GPU 2] Full local | Auditor 1 → est_dp=0.1279 abs_err=0.0349
[Target 4 | GPU 3] Full local | Auditor 1 → est_dp=0.1161 abs_err=0.1140
[Target 3 | GPU 2] Full local | Auditor 2 → est_dp=0.1194 abs_err=0.0434
[Target 2 | GPU 1] Full local | Auditor 3 → est_dp=0.1179 abs_err=0.0105
[Target 4 | GPU 3] Full local | Auditor 2 → est_dp=0.1067 abs_err=0.1234
[Target 1 | GPU 0] Full local | Auditor 3 → est_dp=0.1297 abs_err=0.0335
[Target 4 | GPU 3] Full local | Auditor 3 → est_dp=0.0927 abs_err=0.1374
[Target 3 | GPU 2] Full local | Auditor 4 → est_dp=0.1088 abs_err=0.0540
[Target 2 | GPU 1] Full local | Auditor 4 → est_dp=0.1239 abs_err=0.0045
[Target 1 | GPU 0] Full local | Auditor 4 → est_dp=0.1355 abs_err=0.0277
[Target 4 | GPU 3] Full local | Auditor 5 → est_dp=0.1132 abs_err=0.1169
[Target 4 | GPU 3] Budgeted | Auditor 1 budget=100 → mean_est_dp=0.1282 ± 0.0762 mean_abs_err=0.1115
[Target 4 | GPU 3] Budgeted | Auditor 1 budget=500 → mean_est_dp=0.1107 ± 0.0606 mean_abs_err=0.1194
[Target 3 | GPU 2] Full local | Auditor 5 → est_dp=0.1257 abs_err=0.0371
[Target 3 | GPU 2] Budgeted | Auditor 1 budget=100 → mean_est_dp=0.1855 ± 0.0716 mean_abs_err=0.0546
[Target 2 | GPU 1] Full local | Auditor 5 → est_dp=0.1418 abs_err=0.0133
[Target 1 | GPU 0] Full local | Auditor 5 → est_dp=0.1400 abs_err=0.0232
[Target 2 | GPU 1] Budgeted | Auditor 1 budget=100 → mean_est_dp=0.1470 ± 0.0879 mean_abs_err=0.0777
[Target 1 | GPU 0] Budgeted | Auditor 2 budget=100 → mean_est_dp=0.1520 ± 0.0929 mean_abs_err=0.0740
[Target 3 | GPU 2] Budgeted | Auditor 1 budget=500 → mean_est_dp=0.1530 ± 0.0516 mean_abs_err=0.0445
[Target 4 | GPU 3] Budgeted | Auditor 1 budget=1000 → mean_est_dp=0.1120 ± 0.0419 mean_abs_err=0.1181
[Target 2 | GPU 1] Budgeted | Auditor 1 budget=500 → mean_est_dp=0.1307 ± 0.0922 mean_abs_err=0.0742
[Target 1 | GPU 0] Budgeted | Auditor 2 budget=500 → mean_est_dp=0.1486 ± 0.0363 mean_abs_err=0.0314
[Target 3 | GPU 2] Budgeted | Auditor 1 budget=1000 → mean_est_dp=0.1349 ± 0.0402 mean_abs_err=0.0431
[Target 2 | GPU 1] Budgeted | Auditor 1 budget=1000 → mean_est_dp=0.1417 ± 0.0381 mean_abs_err=0.0352
[Target 1 | GPU 0] Budgeted | Auditor 2 budget=1000 → mean_est_dp=0.1254 ± 0.0410 mean_abs_err=0.0482
[Target 4 | GPU 3] Budgeted | Auditor 1 budget=2000 → mean_est_dp=0.1116 ± 0.0252 mean_abs_err=0.1185
[Target 3 | GPU 2] Budgeted | Auditor 1 budget=2000 → mean_est_dp=0.1360 ± 0.0355 mean_abs_err=0.0385
[Target 2 | GPU 1] Budgeted | Auditor 1 budget=2000 → mean_est_dp=0.1577 ± 0.0321 mean_abs_err=0.0395
[Target 1 | GPU 0] Budgeted | Auditor 2 budget=2000 → mean_est_dp=0.1322 ± 0.0254 mean_abs_err=0.0335
[Target 4 | GPU 3] Budgeted | Auditor 1 budget=5000 → mean_est_dp=0.1158 ± 0.0089 mean_abs_err=0.1143
[Target 4 | GPU 3] Budgeted | Auditor 2 budget=100 → mean_est_dp=0.1235 ± 0.0716 mean_abs_err=0.1067
[Target 4 | GPU 3] Budgeted | Auditor 2 budget=500 → mean_est_dp=0.1076 ± 0.0259 mean_abs_err=0.1225
[Target 3 | GPU 2] Budgeted | Auditor 1 budget=5000 → mean_est_dp=0.1280 ± 0.0170 mean_abs_err=0.0348
[Target 2 | GPU 1] Budgeted | Auditor 1 budget=5000 → mean_est_dp=0.1484 ± 0.0133 mean_abs_err=0.0212
[Target 3 | GPU 2] Budgeted | Auditor 2 budget=100 → mean_est_dp=0.1589 ± 0.0619 mean_abs_err=0.0575
[Target 1 | GPU 0] Budgeted | Auditor 2 budget=5000 → mean_est_dp=0.1312 ± 0.0140 mean_abs_err=0.0320
[Target 2 | GPU 1] Budgeted | Auditor 3 budget=100 → mean_est_dp=0.3521 ± 0.1694 mean_abs_err=0.2340
[Target 1 | GPU 0] Budgeted | Auditor 3 budget=100 → mean_est_dp=0.3563 ± 0.1931 mean_abs_err=0.2461
[Target 3 | GPU 2] Budgeted | Auditor 2 budget=500 → mean_est_dp=0.1286 ± 0.0630 mean_abs_err=0.0613
[Target 4 | GPU 3] Budgeted | Auditor 2 budget=1000 → mean_est_dp=0.1158 ± 0.0340 mean_abs_err=0.1143
[Target 2 | GPU 1] Budgeted | Auditor 3 budget=500 → mean_est_dp=0.1486 ± 0.1171 mean_abs_err=0.1010
[Target 1 | GPU 0] Budgeted | Auditor 3 budget=500 → mean_est_dp=0.1299 ± 0.1168 mean_abs_err=0.1038
[Target 3 | GPU 2] Budgeted | Auditor 2 budget=1000 → mean_est_dp=0.1189 ± 0.0292 mean_abs_err=0.0439
[Target 2 | GPU 1] Budgeted | Auditor 3 budget=1000 → mean_est_dp=0.0998 ± 0.0619 mean_abs_err=0.0572
[Target 1 | GPU 0] Budgeted | Auditor 3 budget=1000 → mean_est_dp=0.1079 ± 0.0543 mean_abs_err=0.0659
[Target 4 | GPU 3] Budgeted | Auditor 2 budget=2000 → mean_est_dp=0.1178 ± 0.0229 mean_abs_err=0.1123
[Target 3 | GPU 2] Budgeted | Auditor 2 budget=2000 → mean_est_dp=0.1113 ± 0.0284 mean_abs_err=0.0528
[Target 2 | GPU 1] Budgeted | Auditor 3 budget=2000 → mean_est_dp=0.1391 ± 0.0479 mean_abs_err=0.0465
[Target 1 | GPU 0] Budgeted | Auditor 3 budget=2000 → mean_est_dp=0.1243 ± 0.0452 mean_abs_err=0.0498
[Target 4 | GPU 3] Budgeted | Auditor 2 budget=5000 → mean_est_dp=0.1077 ± 0.0157 mean_abs_err=0.1224
[Target 4 | GPU 3] Budgeted | Auditor 3 budget=100 → mean_est_dp=0.3315 ± 0.2201 mean_abs_err=0.2094
[Target 4 | GPU 3] Budgeted | Auditor 3 budget=500 → mean_est_dp=0.0932 ± 0.0512 mean_abs_err=0.1369
[Target 3 | GPU 2] Budgeted | Auditor 2 budget=5000 → mean_est_dp=0.1205 ± 0.0206 mean_abs_err=0.0425
[Target 2 | GPU 1] Budgeted | Auditor 3 budget=5000 → mean_est_dp=0.1155 ± 0.0389 mean_abs_err=0.0311
[Target 3 | GPU 2] Budgeted | Auditor 4 budget=100 → mean_est_dp=0.5060 ± 0.0464 mean_abs_err=0.3432
[Target 2 | GPU 1] Budgeted | Auditor 4 budget=100 → mean_est_dp=0.4946 ± 0.0373 mean_abs_err=0.3661
[Target 1 | GPU 0] Budgeted | Auditor 3 budget=5000 → mean_est_dp=0.1414 ± 0.0374 mean_abs_err=0.0319
[Target 1 | GPU 0] Budgeted | Auditor 4 budget=100 → mean_est_dp=0.3881 ± 0.2033 mean_abs_err=0.2779
[Target 3 | GPU 2] Budgeted | Auditor 4 budget=500 → mean_est_dp=0.2726 ± 0.1506 mean_abs_err=0.1433
[Target 4 | GPU 3] Budgeted | Auditor 3 budget=1000 → mean_est_dp=0.1518 ± 0.0623 mean_abs_err=0.0783
[Target 2 | GPU 1] Budgeted | Auditor 4 budget=500 → mean_est_dp=0.2493 ± 0.1903 mean_abs_err=0.1730
[Target 1 | GPU 0] Budgeted | Auditor 4 budget=500 → mean_est_dp=0.2464 ± 0.1805 mean_abs_err=0.1597
[Target 3 | GPU 2] Budgeted | Auditor 4 budget=1000 → mean_est_dp=0.1787 ± 0.1439 mean_abs_err=0.1260
[Target 2 | GPU 1] Budgeted | Auditor 4 budget=1000 → mean_est_dp=0.1542 ± 0.1326 mean_abs_err=0.0988
[Target 1 | GPU 0] Budgeted | Auditor 4 budget=1000 → mean_est_dp=0.1424 ± 0.1312 mean_abs_err=0.1035
[Target 4 | GPU 3] Budgeted | Auditor 3 budget=2000 → mean_est_dp=0.1185 ± 0.0623 mean_abs_err=0.1162
[Target 3 | GPU 2] Budgeted | Auditor 4 budget=2000 → mean_est_dp=0.1555 ± 0.0820 mean_abs_err=0.0715
[Target 2 | GPU 1] Budgeted | Auditor 4 budget=2000 → mean_est_dp=0.1034 ± 0.0687 mean_abs_err=0.0631
[Target 1 | GPU 0] Budgeted | Auditor 4 budget=2000 → mean_est_dp=0.0530 ± 0.0252 mean_abs_err=0.1102
[Target 4 | GPU 3] Budgeted | Auditor 3 budget=5000 → mean_est_dp=0.0919 ± 0.0288 mean_abs_err=0.1382
[Target 4 | GPU 3] Budgeted | Auditor 5 budget=100 → mean_est_dp=0.1429 ± 0.0904 mean_abs_err=0.1023
[Target 4 | GPU 3] Budgeted | Auditor 5 budget=500 → mean_est_dp=0.0595 ± 0.0381 mean_abs_err=0.1706
[Target 3 | GPU 2] Budgeted | Auditor 4 budget=5000 → mean_est_dp=0.1185 ± 0.0555 mean_abs_err=0.0621
[Target 3 | GPU 2] Budgeted | Auditor 5 budget=100 → mean_est_dp=0.2432 ± 0.1231 mean_abs_err=0.1291
[Target 2 | GPU 1] Budgeted | Auditor 4 budget=5000 → mean_est_dp=0.1234 ± 0.0828 mean_abs_err=0.0709
[Target 1 | GPU 0] Budgeted | Auditor 4 budget=5000 → mean_est_dp=0.0848 ± 0.0567 mean_abs_err=0.0827
[Target 2 | GPU 1] Budgeted | Auditor 5 budget=100 → mean_est_dp=0.1664 ± 0.0771 mean_abs_err=0.0624
[Target 1 | GPU 0] Budgeted | Auditor 5 budget=100 → mean_est_dp=0.1942 ± 0.1309 mean_abs_err=0.1126
[Target 3 | GPU 2] Budgeted | Auditor 5 budget=500 → mean_est_dp=0.1610 ± 0.0414 mean_abs_err=0.0329
[Target 4 | GPU 3] Budgeted | Auditor 5 budget=1000 → mean_est_dp=0.1223 ± 0.0527 mean_abs_err=0.1078
[Target 2 | GPU 1] Budgeted | Auditor 5 budget=500 → mean_est_dp=0.1266 ± 0.0617 mean_abs_err=0.0546
[Target 1 | GPU 0] Budgeted | Auditor 5 budget=500 → mean_est_dp=0.1495 ± 0.0686 mean_abs_err=0.0610
[Target 3 | GPU 2] Budgeted | Auditor 5 budget=1000 → mean_est_dp=0.1236 ± 0.0486 mean_abs_err=0.0544
[Target 2 | GPU 1] Budgeted | Auditor 5 budget=1000 → mean_est_dp=0.1245 ± 0.0608 mean_abs_err=0.0479
[Target 1 | GPU 0] Budgeted | Auditor 5 budget=1000 → mean_est_dp=0.1499 ± 0.0530 mean_abs_err=0.0407
[Target 4 | GPU 3] Budgeted | Auditor 5 budget=2000 → mean_est_dp=0.1078 ± 0.0460 mean_abs_err=0.1223
[Target 3 | GPU 2] Budgeted | Auditor 5 budget=2000 → mean_est_dp=0.1211 ± 0.0498 mean_abs_err=0.0596
[Target 2 | GPU 1] Budgeted | Auditor 5 budget=2000 → mean_est_dp=0.1310 ± 0.0301 mean_abs_err=0.0241
[Target 1 | GPU 0] Budgeted | Auditor 5 budget=2000 → mean_est_dp=0.1563 ± 0.0409 mean_abs_err=0.0323
[Target 4 | GPU 3] Budgeted | Auditor 5 budget=5000 → mean_est_dp=0.1047 ± 0.0225 mean_abs_err=0.1254
[Target 3 | GPU 2] Budgeted | Auditor 5 budget=5000 → mean_est_dp=0.1167 ± 0.0194 mean_abs_err=0.0461
[Target 2 | GPU 1] Budgeted | Auditor 5 budget=5000 → mean_est_dp=0.1354 ± 0.0127 mean_abs_err=0.0113
[Target 1 | GPU 0] Budgeted | Auditor 5 budget=5000 → mean_est_dp=0.1410 ± 0.0188 mean_abs_err=0.0229
[Target 4 | GPU 3] Global | est_dp=0.1052 abs_err=0.1249

✓ Target 4 finished (elapsed=1021s)
▶ Launched Target 5 on GPU 0 (pid=36921)
/home/palak/audit/step_5_audit.py:270: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
model.load_state_dict(torch.load(ckpt_path, map_location=device))
[Target 5 | GPU 0] Model loaded from /mnt/nfs/home/palak/data/models/vllm/experiments/lenet_alpha0.5_5nodes_seed42/checkpoints/node_5_best.pt
[Target 3 | GPU 2] Global | est_dp=0.1166 abs_err=0.0462

✓ Target 3 finished (elapsed=1036s)
[Target 1 | GPU 0] Global | est_dp=0.1285 abs_err=0.0347
[Target 2 | GPU 1] Global | est_dp=0.1312 abs_err=0.0027

✓ Target 1 finished (elapsed=1041s)

✓ Target 2 finished (elapsed=1042s)
[Target 5 | GPU 0] Full local | Auditor 1 → est_dp=0.1504 abs_err=0.0112
[Target 5 | GPU 0] Full local | Auditor 2 → est_dp=0.1418 abs_err=0.0198
[Target 5 | GPU 0] Full local | Auditor 3 → est_dp=0.1162 abs_err=0.0454
[Target 5 | GPU 0] Full local | Auditor 4 → est_dp=0.1302 abs_err=0.0314
[Target 5 | GPU 0] Budgeted | Auditor 1 budget=100 → mean_est_dp=0.2040 ± 0.1084 mean_abs_err=0.0960
[Target 5 | GPU 0] Budgeted | Auditor 1 budget=500 → mean_est_dp=0.1439 ± 0.0588 mean_abs_err=0.0453
[Target 5 | GPU 0] Budgeted | Auditor 1 budget=1000 → mean_est_dp=0.1370 ± 0.0483 mean_abs_err=0.0433
[Target 5 | GPU 0] Budgeted | Auditor 1 budget=2000 → mean_est_dp=0.1336 ± 0.0197 mean_abs_err=0.0283
[Target 5 | GPU 0] Budgeted | Auditor 1 budget=5000 → mean_est_dp=0.1495 ± 0.0143 mean_abs_err=0.0156
[Target 5 | GPU 0] Budgeted | Auditor 2 budget=100 → mean_est_dp=0.1102 ± 0.0627 mean_abs_err=0.0728
[Target 5 | GPU 0] Budgeted | Auditor 2 budget=500 → mean_est_dp=0.1734 ± 0.0514 mean_abs_err=0.0446
[Target 5 | GPU 0] Budgeted | Auditor 2 budget=1000 → mean_est_dp=0.1450 ± 0.0367 mean_abs_err=0.0308
[Target 5 | GPU 0] Budgeted | Auditor 2 budget=2000 → mean_est_dp=0.1373 ± 0.0253 mean_abs_err=0.0305
[Target 5 | GPU 0] Budgeted | Auditor 2 budget=5000 → mean_est_dp=0.1371 ± 0.0134 mean_abs_err=0.0245
[Target 5 | GPU 0] Budgeted | Auditor 3 budget=100 → mean_est_dp=0.2560 ± 0.2340 mean_abs_err=0.2025
[Target 5 | GPU 0] Budgeted | Auditor 3 budget=500 → mean_est_dp=0.1742 ± 0.1118 mean_abs_err=0.0876
[Target 5 | GPU 0] Budgeted | Auditor 3 budget=1000 → mean_est_dp=0.1140 ± 0.0703 mean_abs_err=0.0718
[Target 5 | GPU 0] Budgeted | Auditor 3 budget=2000 → mean_est_dp=0.0991 ± 0.0574 mean_abs_err=0.0719
[Target 5 | GPU 0] Budgeted | Auditor 3 budget=5000 → mean_est_dp=0.1068 ± 0.0355 mean_abs_err=0.0587
[Target 5 | GPU 0] Budgeted | Auditor 4 budget=100 → mean_est_dp=0.5010 ± 0.0386 mean_abs_err=0.3393
[Target 5 | GPU 0] Budgeted | Auditor 4 budget=500 → mean_est_dp=0.3158 ± 0.2004 mean_abs_err=0.2128
[Target 5 | GPU 0] Budgeted | Auditor 4 budget=1000 → mean_est_dp=0.2233 ± 0.1394 mean_abs_err=0.1114
[Target 5 | GPU 0] Budgeted | Auditor 4 budget=2000 → mean_est_dp=0.1603 ± 0.0811 mean_abs_err=0.0635
[Target 5 | GPU 0] Budgeted | Auditor 4 budget=5000 → mean_est_dp=0.1400 ± 0.0740 mean_abs_err=0.0670
[Target 5 | GPU 0] Global | est_dp=0.1381 abs_err=0.0235

✓ Target 5 finished (elapsed=2061s)

All targets finished in 2061s (34.3 min)

──────────────────────────────────────────────────────────────────────
Full Local Audit — Summary
──────────────────────────────────────────────────────────────────────

Auditor → Target Est DP Err(data) Err(mdl_val) Err(mdl_full)

---

Node 1 → Node 2 0.1393 0.0012 0.0108 0.0043
Node 1 → Node 3 0.1279 0.0050 0.0349 0.0139
Node 1 → Node 4 0.1161 0.0199 0.1140 0.0117
Node 1 → Node 5 0.1504 0.0000 0.0112 0.0017
Node 2 → Node 1 0.1312 0.0136 0.0320 0.0050
Node 2 → Node 3 0.1194 0.0135 0.0434 0.0054
Node 2 → Node 4 0.1067 0.0294 0.1234 0.0023
Node 2 → Node 5 0.1418 0.0086 0.0198 0.0069
Node 3 → Node 1 0.1297 0.0151 0.0335 0.0065
Node 3 → Node 2 0.1179 0.0226 0.0105 0.0171
Node 3 → Node 4 0.0927 0.0433 0.1374 0.0117
Node 3 → Node 5 0.1162 0.0342 0.0454 0.0325
Node 4 → Node 1 0.1355 0.0093 0.0277 0.0007
Node 4 → Node 2 0.1239 0.0165 0.0045 0.0110
Node 4 → Node 3 0.1088 0.0242 0.0540 0.0053
Node 4 → Node 5 0.1302 0.0202 0.0314 0.0185
Node 5 → Node 1 0.1400 0.0048 0.0232 0.0038
Node 5 → Node 2 0.1418 0.0013 0.0133 0.0068
Node 5 → Node 3 0.1257 0.0072 0.0371 0.0117
Node 5 → Node 4 0.1132 0.0229 0.1169 0.0087
[Data] abs err: 0.0156 ± 0.0113 rel err: 11.2% ± 8.2%
[Model Val] abs err: 0.0462 ± 0.0405 rel err: 24.3% ± 16.4%
[Model Full] abs err: 0.0093 ± 0.0072 rel err: 7.3% ± 5.1%

──────────────────────────────────────────────────────────────────────
Global Audit — Summary
──────────────────────────────────────────────────────────────────────

Global → Node 1 est_dp=0.1285 err(data)=0.0163 err(mdl_val)=0.0347 err(mdl_full)=0.0077
Global → Node 2 est_dp=0.1312 err(data)=0.0093 err(mdl_val)=0.0027 err(mdl_full)=0.0038
Global → Node 3 est_dp=0.1166 err(data)=0.0163 err(mdl_val)=0.0462 err(mdl_full)=0.0026
Global → Node 4 est_dp=0.1052 err(data)=0.0308 err(mdl_val)=0.1249 err(mdl_full)=0.0008
Global → Node 5 est_dp=0.1381 err(data)=0.0123 err(mdl_val)=0.0235 err(mdl_full)=0.0106

──────────────────────────────────────────────────────────────────────
Generating Plots
──────────────────────────────────────────────────────────────────────

✓ /mnt/nfs/home/palak/data/models/vllm/experiments/lenet_alpha0.5_5nodes_seed42/plots/step5_full_estimated_vs_true.png
✓ /mnt/nfs/home/palak/data/models/vllm/experiments/lenet_alpha0.5_5nodes_seed42/plots/step5_full_error_heatmap.png
✓ /mnt/nfs/home/palak/data/models/vllm/experiments/lenet_alpha0.5_5nodes_seed42/plots/step5_budget_sample_efficiency.png
✓ /mnt/nfs/home/palak/data/models/vllm/experiments/lenet_alpha0.5_5nodes_seed42/plots/step5_budget_error_vs_mismatch.png
✓ /mnt/nfs/home/palak/data/models/vllm/experiments/lenet_alpha0.5_5nodes_seed42/plots/step5_global_vs_local.png
✓ /mnt/nfs/home/palak/data/models/vllm/experiments/lenet_alpha0.5_5nodes_seed42/plots/step5_ranking_accuracy.png

✓ Results saved → /mnt/nfs/home/palak/data/models/vllm/experiments/lenet_alpha0.5_5nodes_seed42/results/step5_audit_results.json

======================================================================
Step 5 Complete
======================================================================

Total audits run : 1,025
Total time : 2061s (34.3 min)
Full local abs err : 0.0462 ± 0.0405
Full local rel err : 24.3% ± 16.4%

Outputs saved to: /mnt/nfs/home/palak/data/models/vllm/experiments/lenet_alpha0.5_5nodes_seed42
plots/step5_full_estimated_vs_true.png
plots/step5_full_error_heatmap.png
plots/step5_budget_sample_efficiency.png
plots/step5_budget_error_vs_mismatch.png
plots/step5_global_vs_local.png
plots/step5_ranking_accuracy.png
results/step5_audit_results.json
logs/step5.log
======================================================================
palak@audit-4-nodes-0-0:~/audit$
