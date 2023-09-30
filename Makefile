Train_r18:
	python train.py --data-dir ../../dataset/ \
                --dataset cv/ --train-split \
                train --val-split validation \
                --num-classes 43 --model resnet18 --experiment Train_r18\
                --img-size 32 --no-aug --no-prefetcher --batch-size 512


Train_r18_focal:
	python train.py --data-dir ../../dataset/ \
                --dataset cv/ --train-split \
                train --val-split validation \
                --num-classes 43 --model resnet18 --experiment Train_r18_focal\
                --img-size 32 --no-aug --no-prefetcher --batch-size 512 --focal

Train_r18_focal_4:
	python train.py --data-dir ../../dataset/ \
                --dataset cv/ --train-split \
                train --val-split validation \
                --num-classes 43 --model resnet18 --experiment Train_r18_focal_4\
                --img-size 32 --no-aug --no-prefetcher --batch-size 512 --focal

Train_r18_focal_8:
	python train.py --data-dir ../../dataset/ \
                --dataset cv/ --train-split \
                train --val-split validation \
                --num-classes 43 --model resnet18 --experiment Train_r18_focal_8\
                --img-size 32 --no-aug --no-prefetcher --batch-size 512 --focal

Train_r50:
	python train.py --data-dir ../../dataset/ \
                --dataset cv/ --train-split \
                train --val-split validation \
                --num-classes 43 --model resnet50 \
                --img-size 32 --no-aug --no-prefetcher --batch-size 512

Train_nxt_pico:
	python train.py --data-dir ../../dataset/ \
                --dataset cv/ --train-split \
                train --val-split validation \
                --num-classes 43 --model convnextv2_pico \
                --img-size 32 --no-aug --no-prefetcher --batch-size 512

Train_nxt_pico_aug:
	python train.py --data-dir ../../dataset/ \
                --dataset cv/ --train-split \
                train --val-split validation \
                --num-classes 43 --model convnextv2_pico \
                --img-size 32 --no-prefetcher --batch-size 512 \
				--hflip 0 --aa original

Train_nxt_pico_ema:
	python train.py --data-dir ../../dataset/ \
                --dataset cv/ --train-split \
                train --val-split validation \
                --num-classes 43 --model convnextv2_pico \
                --img-size 32 --no-aug --no-prefetcher --batch-size 512 --model-ema

Train_efficientnetv2_s:
	python train.py --data-dir ../../dataset/ \
                --dataset cv/ --train-split \
                train --val-split validation \
                --num-classes 43 --model efficientnetv2_s \
                --img-size 32 --no-aug --no-prefetcher --batch-size 256

Train_resnet18_pret:
	python train.py --data-dir ../../dataset/ \
                --dataset cv/ --train-split \
                train --val-split validation \
                --num-classes 43 --model resnet18 --experiment Train_resnet18_pret \
                --img-size 32 --no-aug --no-prefetcher --batch-size 256 --pretrained --lr 0.0001

Train_resnet18_pret_224:
	python train.py --data-dir ../../dataset/ \
                --dataset cv/ --train-split \
                train --val-split validation \
                --num-classes 43 --model resnet18 --experiment Train_resnet18_pret_224 \
                --img-size 224 --no-aug --no-prefetcher --batch-size 32 --pretrained

Train_nxt_pico_pret_224:
	python train.py --data-dir ../../dataset/ \
                --dataset cv/ --train-split \
                train --val-split validation \
                --num-classes 43 --model convnextv2_pico --experiment Train_nxt_pico_pret_224 \
                --img-size 224 --no-aug --no-prefetcher --batch-size 24 --pretrained

Train_nxt_pico_224:
	python train.py --data-dir ../../dataset/ \
                --dataset cv/ --train-split \
                train --val-split validation \
                --num-classes 43 --model convnextv2_pico --experiment Train_nxt_pico_224 \
                --img-size 224 --no-aug --no-prefetcher --batch-size 24

Train_resnet18_pret_224_25ep:
	python train.py --data-dir ../../dataset/ \
                --dataset cv/ --train-split \
                train --val-split validation \
                --num-classes 43 --model resnet18 --experiment Train_resnet18_pret_224_25ep \
                --img-size 224 --no-aug --no-prefetcher --batch-size 32 --pretrained --epochs 25


Train_resnet152_pret:
	python train.py --data-dir ../../dataset/ \
                --dataset cv/ --train-split \
                train --val-split validation \
                --num-classes 43 --model resnet152 --experiment Train_resnet152_pret \
                --img-size 32 --no-aug --no-prefetcher --batch-size 64 --pretrained --lr 0.0001

Train_nxt_pico_pret:
	python train.py --data-dir ../../dataset/ \
                --dataset cv/ --train-split \
                train --val-split validation \
                --num-classes 43 --model convnextv2_pico --experiment Train_nxt_pico_pret \
                --img-size 32 --no-aug --no-prefetcher --batch-size 256 --sched plateau \
				#--pretrained

Train_efficientnetv2_s_aa:
	python train.py --data-dir ../../dataset/ \
                --dataset cv/ --train-split \
                train --val-split validation \
                --num-classes 43 --model efficientnetv2_s \
                --img-size 32 --no-prefetcher --batch-size 256 \
				--hflip 0 --aa original --experiment Train_efficientnetv2_s_aa


Train_efficientnetv2_s_mixup:
	python train.py --data-dir ../../dataset/ \
                --dataset cv/ --train-split \
                train --val-split validation \
                --num-classes 43 --model efficientnetv2_s \
                --img-size 32 --no-aug --no-prefetcher --batch-size 256 \
				--hflip 0 --experiment Train_efficientnetv2_s_mixup --mixup 0.6

Train_efficientnetv2_s_mixup_100ep:
	python train.py --data-dir ../../dataset/ \
                --dataset cv/ --train-split \
                train --val-split validation \
                --num-classes 43 --model efficientnetv2_s \
                --img-size 32 --no-aug --no-prefetcher --batch-size 256 \
				--hflip 0 --experiment Train_efficientnetv2_s_mixup_100ep --mixup 0.6 --epochs 100