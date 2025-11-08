pip install --upgrade huggingface_hub[hf_transfer]

hf download --repo-type dataset Xkev/LLaVA-CoT-100k \
            --local-dir /scratch/tjgus0408/huggingface/datasets/LLaVA-CoT-100k 