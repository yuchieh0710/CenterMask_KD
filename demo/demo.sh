python3 ./centermask_demo.py --weights '../tools/checkpoints/student/model_0360000.pth' --output_dir 'demo/results/student'

#python3 ./centermask_demo.py --weights '../tools/checkpoints/teacher/model_0360000.pth' --output_dir 'demo/results/teacher'

python3 ./centermask_demo.py --weights '../tools/checkpoints/kd_corr_KL/model_0360000.pth' --output_dir 'demo/results/logits(KL) Ocorr'

python3 ./centermask_demo.py --weights '../tools/checkpoints/kd_head/model_0360000.pth' --output_dir 'demo/results/logits(CE)'

python3 ./centermask_demo.py --weights '../tools/checkpoints/kd_head_KL/model_0360000.pth' --output_dir 'demo/results/logits(KL)'

python3 ./centermask_demo.py --weights '../tools/checkpoints/kd_learn_att_corr/model_0360000.pth' --output_dir 'demo/results/logits(CE) Ocorr'

python3 ./centermask_demo.py --weights '../tools/checkpoints/kd_learn_att_MSE/model_0360000.pth' --output_dir 'demo/results/logits(CE) Xcorr'

python3 ./centermask_demo.py --weights '../tools/checkpoints/kd_MSE_KL/model_0360000.pth' --output_dir 'demo/results/logits(KL) Xcorr'

