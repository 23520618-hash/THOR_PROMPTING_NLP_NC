import os
import pandas as pd
import openai
import time
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report
from collections import defaultdict

# Import cho OpenAI package mới
try:
    from openai import OpenAI, RateLimitError, APIError, APIConnectionError
except ImportError:
    from openai.error import RateLimitError, APIError, APIConnectionError

import backoff

# ========== CẤU HÌNH TRỰC TIẾP ==========
API_KEY = "OpenAi-key"
CSV_FILE = "D:\\Nghiên cứu khoa học\\conference_4\\THOR-NEW\\implicit_sentiment_laptop.csv"
MODEL_NAME = "gpt-5.1"
API_ENDPOINT = ""  # Để trống nếu dùng OpenAI mặc định
# =======================================

# ===== TIMING & TOKEN METRICS =====
TOTAL_PROCESSING_TIME = 0.0
TOTAL_INFERENCE_TIME = 0.0

TOTAL_INPUT_TOKENS = 0
TOTAL_OUTPUT_TOKENS = 0
TOTAL_COMPLETION_TOKENS = 0

TOTAL_API_CALLS = 0

def prompt_for_aspect_inferring(context, target):
    new_context = f'Given the sentence "{context}", '
    prompt = new_context + f'which specific aspect of {target} is possibly mentioned?'
    return new_context, prompt

def prompt_for_opinion_inferring(context, target, aspect_expr):
    new_context = context + ' ' + aspect_expr
    prompt = new_context + f' Based on the common sense, what is the implicit opinion towards the mentioned aspect of {target}, and why?'
    return new_context, prompt

def prompt_for_polarity_inferring(context, target, opinion_expr):
    new_context = context + ' ' + opinion_expr
    prompt = new_context + f' Based on such opinion, what is the sentiment polarity towards {target}?'
    return new_context, prompt

def prompt_for_polarity_label(context, polarity_expr):
    prompt = polarity_expr + ' Based on these contexts, summarize the sentiment polarity, and return only one of these words: positive, neutral, or negative.'
    return prompt

def load_csv_data(csv_file_path):
    df = pd.read_csv(csv_file_path)
    required_columns = ['text', 'target', 'label']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Thiếu cột '{col}' trong file CSV")

    data = []
    for _, row in df.iterrows():
        text = str(row['text'])
        target = str(row['target'])
        label = row['label']

        if isinstance(label, str):
            label = label.lower().strip()
            label_mapping = {'positive': 0, 'negative': 1, 'neutral': 2,
                             'pos': 0, 'neg': 1, 'neu': 2}
            label = label_mapping.get(label, 2)

        data.append([text, target, label])

    print(f"Đã tải {len(data)} mẫu từ file: {csv_file_path}")
    return data

@backoff.on_exception(backoff.expo, (RateLimitError, APIError, APIConnectionError))
def request_result(conversation, prompt_text, model_name, api_key):
    global TOTAL_INFERENCE_TIME
    global TOTAL_INPUT_TOKENS, TOTAL_OUTPUT_TOKENS, TOTAL_COMPLETION_TOKENS
    global TOTAL_API_CALLS

    conversation.append({'role': 'user', "content": prompt_text})

    infer_start = time.time()

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model_name,
        messages=conversation,
    )

    infer_time = time.time() - infer_start
    TOTAL_INFERENCE_TIME += infer_time
    TOTAL_API_CALLS += 1

    if response.usage:
        TOTAL_INPUT_TOKENS += response.usage.prompt_tokens
        TOTAL_OUTPUT_TOKENS += response.usage.completion_tokens
        TOTAL_COMPLETION_TOKENS += response.usage.total_tokens

    result = response.choices[0].message.content.replace('\n', ' ').strip()

    conversation.append({
        "role": "assistant",
        "content": result
    })

    return conversation, result

def evaluate_sentiment():
    global TOTAL_PROCESSING_TIME   # ✅ FIX LỖI Ở ĐÂY

    data = load_csv_data(CSV_FILE)

    system_role = {
        'role': 'system',
        'content': "You are an expert at analyzing implicit sentiments and opinions."
    }

    label_list = ['positive', 'negative', 'neutral']

    all_predictions = []
    all_gold_labels = []

    counter_file = 'counter.txt'
    if os.path.exists(counter_file):
        os.remove(counter_file)

    for i, (sent, target, gold_label) in enumerate(tqdm(data)):
        row_start = time.time()

        conversation = [system_role]

        try:
            context_step1, step_1_prompt = prompt_for_aspect_inferring(sent, target)
            conversation, aspect_expr = request_result(conversation, step_1_prompt, MODEL_NAME, API_KEY)

            context_step2, step_2_prompt = prompt_for_opinion_inferring(context_step1, target, aspect_expr)
            conversation, opinion_expr = request_result(conversation, step_2_prompt, MODEL_NAME, API_KEY)

            context_step3, step_3_prompt = prompt_for_polarity_inferring(context_step2, target, opinion_expr)
            conversation, polarity_expr = request_result(conversation, step_3_prompt, MODEL_NAME, API_KEY)

            step_lb_prompt = prompt_for_polarity_label(context_step3, polarity_expr)
            conversation, final_output = request_result(conversation, step_lb_prompt, MODEL_NAME, API_KEY)

            final_output = final_output.lower().strip()
            predicted_label = 2
            for idx, label in enumerate(label_list):
                if label in final_output:
                    predicted_label = idx
                    break

            all_predictions.append(predicted_label)
            all_gold_labels.append(gold_label)

        except Exception:
            all_predictions.append(2)
            all_gold_labels.append(gold_label)

        TOTAL_PROCESSING_TIME += time.time() - row_start

    accuracy = accuracy_score(all_gold_labels, all_predictions)
    f1_macro = f1_score(all_gold_labels, all_predictions, average='macro')

    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"F1-Macro: {f1_macro:.4f}")
    print(classification_report(all_gold_labels, all_predictions, target_names=label_list))

    print("\n⏱️ TIMING METRICS:")
    print(f"Total Processing Time: {TOTAL_PROCESSING_TIME:.2f} seconds")
    print(f"Total Inference Time: {TOTAL_INFERENCE_TIME:.2f} seconds")
    print(f"Avg Inference per API call: {(TOTAL_INFERENCE_TIME / TOTAL_API_CALLS) * 1000:.2f} ms")
    print(f"Avg Processing per review: {(TOTAL_PROCESSING_TIME / TOTAL_API_CALLS) * 1000:.2f} ms")

    print("\n🔤 TOKEN USAGE:")
    print(f"Total Input Tokens: {TOTAL_INPUT_TOKENS}")
    print(f"Total Output Tokens: {TOTAL_OUTPUT_TOKENS}")
    print(f"Total Completion Tokens: {TOTAL_COMPLETION_TOKENS}")
    print(f"Avg Input per API call: {TOTAL_INPUT_TOKENS / TOTAL_API_CALLS:.2f}")
    print(f"Avg Output per API call: {TOTAL_OUTPUT_TOKENS / TOTAL_API_CALLS:.2f}")

if __name__ == '__main__':
    evaluate_sentiment()
