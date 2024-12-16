python generate.py \
    --disciplines_filepath disciplines1.txt \
    --language Chinese \
    --max_number_of_subjects 15 \
    --max_number_of_subtopics 30 \
    --max_number_of_session_name 30 \
    --num_iterations 15 \
    --num_questions_per_iteration 18 \
    --question_max_tokens 1024 \
    --question_batch_size 9 \
    --model_name_for_answer gpt-4o \
    --answer_max_tokens 2048 \
    --answer_batch_size 9