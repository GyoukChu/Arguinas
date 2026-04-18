"""
Do ctrl+f and search "###" for the parts to modify each time!
"""

import re
import json
import time
import argparse
import subprocess
from subprocess import check_output
from tqdm import tqdm
import os
from os.path import join
import openai
import anthropic
from dotenv import load_dotenv

# Load API keys from a local .env file (see .env.example).
# Keys live in ANTHROPIC_API_KEY / OPENAI_API_KEY.
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))
if not os.environ.get("ANTHROPIC_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
    raise RuntimeError(
        "No API key found. Set ANTHROPIC_API_KEY and/or OPENAI_API_KEY in a .env file."
    )
###
RECON_FORMAT = """Your output should be composed of two parts: argument reconstruction and its formalization. 
 - In the first part, list premises, intermediate conclusions, and the conclusion, and indicate their logical connections (i.e., which propositions deductively imply which).
 - In the second part, first define variables and/or predicates, then formalize premises, intermediate conclusions, and a conclusion, and then generate a deductive proof.
The output format should be as follows.

# Argument Reconstruction

## Premises
[list of explicit and implicit premises]

## Intermediate Conclusions
[list of intermediate conclusions (if intermediate conclusions are not needed, then write "None".)]

## Conclusion
[a conclusion]

## Logical Connections
[list of logical connections]

# Formalized Argument

## Defined Variables/Predicates
[definition of each variable and/or predicate]

## Formalized Premises
[formalization of premises using definition]

## Formalized Intermediate Conclusions
[formalization of intermediate conclusions using definition (if intermediate conclusions are not needed, then write "None".)]

## Formalized Conclusion
[formalization of conclusion using definition]

## Deductive Proof
[deductive proof using formalized premises]"""

RECON_FORMAT_FORMAL_FALLACY = """Your output should be composed of two parts: argument reconstruction and its formalization. 
 - In the first part, list premises, intermediate conclusions, and the conclusion.
 - In the second part, first define variables and/or predicates, then formalize premises, intermediate conclusions, and a conclusion.
The output format should be as follows.

# Argument Reconstruction

## Premises
[list of explicit and implicit premises]

## Intermediate Conclusions
[list of intermediate conclusions (if intermediate conclusions are not needed, then write "None".)]

## Conclusion
[a conclusion]

# Formalized Argument

## Defined Variables/Predicates
[definition of each variable and/or predicate]

## Formalized Premises
[formalization of premises using definition]

## Formalized Intermediate Conclusions
[formalization of intermediate conclusions using definition (if intermediate conclusions are not needed, then write "None".)]

## Formalized Conclusion
[formalization of conclusion using definition]"""


class LogicalStreamliner:

    def __init__(self, args):
        self.args= args
        self.data_path = args.data_path
        self.data_filename = args.data_filename
        self.use_general_reconstruction = args.use_general_reconstruction
        self.use_specific_reconstruction = args.use_specific_reconstruction
        self.save_path = args.save_path
        self.prompt_path = args.prompt_path
        self.model = args.model_name
        self.subset = args.subset
        self.max_num_debug = args.max_num_debug
        self.max_num_recon = args.max_num_recon
        self.max_attempts = args.max_attempts
        self.load_prompt_templates()

        # create the folder to save the program
        cache_dir = os.path.join(os.path.dirname(__file__), '.cache_program')
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        self.cache_dir = cache_dir
    
    def load_prompt_templates(self): ###

        with open(os.path.join(self.prompt_path, "fallacy_detection.txt"), 'r') as f:
            self.prompt_fallacy_detection = f.read()
        # General reconstruction prompts
        if self.use_general_reconstruction:
            with open(os.path.join(self.prompt_path, "reconstruction_general_no_fallacy.txt"), 'r') as f:
                self.prompt_reconstruction_no_fallacy = f.read()
            with open(os.path.join(self.prompt_path, "reconstruction_general_formal_fallacy.txt"), 'r') as f:
                self.prompt_reconstruction_formal_fallacy = f.read()
            with open(os.path.join(self.prompt_path, "reconstruction_general_informal_fallacy.txt"), 'r') as f:
                self.prompt_reconstruction_informal_fallacy = f.read()
            with open(os.path.join(self.prompt_path, "reconstruction_general_both_fallacy.txt"), 'r') as f:
                self.prompt_reconstruction_both_fallacy = f.read()
        # Specific reconstruction prompts
        if self.use_specific_reconstruction:
            with open(os.path.join(self.prompt_path, "reconstruction_60_types_no_fallacy.txt"), 'r') as f:
                self.prompt_reconstruction_no_fallacy = f.read()
            with open(os.path.join(self.prompt_path, "reconstruction_60_types_formal_fallacy.txt"), 'r') as f:
                self.prompt_reconstruction_formal_fallacy = f.read()
            with open(os.path.join(self.prompt_path, "reconstruction_60_types_informal_fallacy.txt"), 'r') as f:
                self.prompt_reconstruction_informal_fallacy = f.read()
            with open(os.path.join(self.prompt_path, "reconstruction_60_types_both_fallacy.txt"), 'r') as f:
                self.prompt_reconstruction_both_fallacy = f.read()
        
        with open(os.path.join(self.prompt_path, "check_validity.txt"), 'r') as f:
            self.prompt_validity = f.read()
        with open(os.path.join(self.prompt_path, "check_faithfulness.txt"), 'r') as f:
            self.prompt_faithfulness = f.read()
        with open(os.path.join(self.prompt_path, "streamlining.txt"), 'r') as f:
            self.prompt_deformalization = f.read()
        with open(os.path.join(self.prompt_path, "program_debugging.txt"), 'r') as f:
            self.prompt_debug = f.read()
        

    def load_raw_dataset(self):
        # load json files
        data_file = os.path.join(self.data_path, f'{self.data_filename}')
        with open(data_file) as f:
            dataset = json.load(f)
        return dataset
    
    def count_price(self, total_input_tokens, total_output_tokens):

        MODEL_PRICE_DICT = {
            "claude-sonnet-4-6": (3, 15),
            "claude-sonnet-4-5-20250929": (3, 15),
            "claude-haiku-4-5-20251001": (1, 5),
            "claude-opus-4-5-20251101": (5, 25),
            "claude-sonnet-4-20250514": (3, 15),
            # "claude-3-7-sonnet-20250219": (3, 15), -- deprecated
            "gpt-5.1-2025-11-13": (1.25, 10),
            "gpt-5-2025-08-07": (1.25, 10),
            "gpt-5-mini-2025-08-07": (0.25, 2),
            "gpt-4.1-2025-04-14": (2, 8),
            "gpt-4.1-mini-2025-04-14": (0.4, 1.6),
            "gpt-4o-2024-08-06": (2.5, 10),
            "gpt-4o-mini-2024-07-18": (0.15, 0.6),
        }
        if self.model not in MODEL_PRICE_DICT:
            raise ValueError(f'model unsupported: {self.model}')
        
        input_price, output_price = MODEL_PRICE_DICT[self.model]
        total_price = (input_price * total_input_tokens + output_price * total_output_tokens) / 1000000 # Cost per 1M tokens
        return total_price

    def api_call(self, input_, max_tokens=8192, temperature=1.0, thinking=False, reasoning_effort="minimal"):
        MAX_RETRIES = 10
        RETRY_DELAY = 60  # seconds
        
        if type(input_) == list:
            messages = []
            for i in range(len(input_)):
                if i % 2 == 0:
                    role = 'user'
                else:
                    role = 'assistant'
                messages.append({"role": role, "content": input_[i]})
        else:
            messages = [{"role": "user", "content": input_}]

        last_exception = None
        for retry_count in range(MAX_RETRIES):
            try:
                if self.model in ["claude-sonnet-4-6", "claude-sonnet-4-5-20250929", "claude-sonnet-4-20250514", "claude-haiku-4-5-20251001"]: # "claude-3-7-sonnet-20250219" deprecated
                    if not thinking:
                        message = anthropic.Anthropic().messages.create(
                            model=self.model,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            messages=messages
                        )
                        output_ = message.content[0].text
                        input_tokens_ = message.usage.input_tokens
                        output_tokens_ = message.usage.output_tokens
                    else:
                        message = anthropic.Anthropic().messages.create(
                            model=self.model,
                            max_tokens=max_tokens * 2,
                            thinking={
                                "type": "enabled",
                                "budget_tokens": max(1024, int(max_tokens * 2 * 0.7))
                            },
                            temperature=temperature,
                            messages=messages
                        )
                        output_ = "".join(b.text for b in message.content if getattr(b, "type", None) == "text")
                        input_tokens_ = message.usage.input_tokens
                        output_tokens_ = message.usage.output_tokens

                elif self.model in ["gpt-5.1-2025-11-13", "gpt-5-2025-08-07", "gpt-5-mini-2025-08-07"]:
                    if self.model in ["gpt-5.1-2025-11-13"] and reasoning_effort == "minimal":
                        reasoning_effort = "none" # gpt 5.1 is "none", not "minimal"
                    
                    response = openai.OpenAI().responses.create(
                        model=self.model,
                        input=messages,
                        text={
                            "format": {
                                "type": "text"
                            },
                            "verbosity": "medium" ## Default value
                        },
                        reasoning={
                            "effort": reasoning_effort
                        },
                        tools=[],
                        max_output_tokens=max_tokens * 2,
                    )
                    output_ = response.output_text.strip()
                    input_tokens_ = response.usage.input_tokens
                    output_tokens_ = response.usage.output_tokens

                elif self.model in ["gpt-4.1-2025-04-14", "gpt-4.1-mini-2025-04-14", "gpt-4o-2024-08-06", "gpt-4o-mini-2024-07-18"]:
                    # No reasoning
                    response = openai.OpenAI().responses.create(
                        model=self.model,
                        input=messages,
                        max_output_tokens=max_tokens,
                        temperature=temperature,
                    )
                    output_ = response.output_text.strip()
                    input_tokens_ = response.usage.input_tokens
                    output_tokens_ = response.usage.output_tokens

                else:
                    raise ValueError(f'model unsupported: {self.model}')

                return output_, input_tokens_, output_tokens_
            
            except ValueError:
                # ValueError (unsupported model) should not be retried
                raise
            except Exception as e:
                last_exception = e
                print(f"[API Error] Attempt {retry_count + 1}/{MAX_RETRIES} failed: {type(e).__name__}: {e}")
                if retry_count < MAX_RETRIES - 1:
                    print(f"[API Error] Retrying in {RETRY_DELAY} seconds...")
                    time.sleep(RETRY_DELAY)
        
        # All retries failed
        raise RuntimeError(f"API call failed after {MAX_RETRIES} attempts. Last error: {last_exception}")

    def detect_fallacy(self, topic, background, argument):
        input_ = self.prompt_fallacy_detection.replace('[[Topic]]', topic).replace('[[Background]]', background).replace('[[ARG]]', argument)

        attempt = 0
        while attempt < self.max_attempts:
            output_, input_tokens_, output_tokens_ = self.api_call(input_, temperature=1.0)
            print("<| Detect Fallacy |>")
            print(output_)

            pattern = r"# Reasoning\n(.*?)" \
            r"# Formal Fallacy\n(.*?)" \
            r"## Rationale of Formal Fallacy\n(.*?)" \
            r"# Informal Fallacy\n(.*?)" \
            r"## Rationale of Informal Fallacy\n(.*?)\Z"
            
            match = re.search(pattern, output_, re.DOTALL)
            
            if match:
                sections_fallacy_detection = {
                    "reasoning": match.group(1).strip(),
                    "formal_fallacy": match.group(2).strip(),
                    "formal_fallacy_rationale": match.group(3).strip(),
                    "informal_fallacy": match.group(4).strip(),
                    "informal_fallacy_rationale": match.group(5).strip()
                }
                # check if the formal fallacy is yes or no
                if 'yes' in sections_fallacy_detection['formal_fallacy'].strip().lower():
                    is_formal_fallacy = True
                elif 'no' in sections_fallacy_detection['formal_fallacy'].strip().lower():
                    is_formal_fallacy = False
                else: # Retry
                    attempt += 1
                    continue
                # check if the informal fallacy is yes or no
                if 'yes' in sections_fallacy_detection['informal_fallacy'].strip().lower():
                    is_informal_fallacy = True
                elif 'no' in sections_fallacy_detection['informal_fallacy'].strip().lower():
                    is_informal_fallacy = False
                else: # Retry
                    attempt += 1
                    continue

                return is_formal_fallacy, is_informal_fallacy, sections_fallacy_detection, input_tokens_, output_tokens_
            
            attempt += 1
        
        raise ValueError(f'raw output:\n{output_}')

    def generate_reconstruction(self, formal_fallacy_flag, informal_fallacy_flag, sections_fallacy_detection, topic, background, argument, message_list, feedback_faithfulness=None, sections_final=None):
        if len(message_list) == 0: # first reconstruction
            if formal_fallacy_flag and informal_fallacy_flag:
                rationale_formal = sections_fallacy_detection['formal_fallacy_rationale']
                rationale_informal = sections_fallacy_detection['informal_fallacy_rationale']
                input_ = self.prompt_reconstruction_both_fallacy.replace('[[Topic]]', topic).replace('[[Background]]', background).replace('[[ARG]]', argument).replace('[[RATIONALE_FORMAL]]', rationale_formal).replace('[[RATIONALE_INFORMAL]]', rationale_informal)
            elif formal_fallacy_flag:
                rationale_formal = sections_fallacy_detection['formal_fallacy_rationale']
                input_ = self.prompt_reconstruction_formal_fallacy.replace('[[Topic]]', topic).replace('[[Background]]', background).replace('[[ARG]]', argument).replace('[[RATIONALE_FORMAL]]', rationale_formal)
            elif informal_fallacy_flag:
                rationale_informal = sections_fallacy_detection['informal_fallacy_rationale']
                input_ = self.prompt_reconstruction_informal_fallacy.replace('[[Topic]]', topic).replace('[[Background]]', background).replace('[[ARG]]', argument).replace('[[RATIONALE_INFORMAL]]', rationale_informal)
            else:
                input_ = self.prompt_reconstruction_no_fallacy.replace('[[Topic]]', topic).replace('[[Background]]', background).replace('[[ARG]]', argument)

        elif feedback_faithfulness is not None: # faithfulness f/b
            input_ = f'Using the formalized argument, we could generate a more clarified argument reconstruction, as shown below.\n\n# Argument Reconstruction\n## Premises\n[[PREMISES]]\n\n## Conclusion\n[[CONCLUSION]]\n\nHowever, this reconstruction is NOT faithful for the following reason.\n\n# Feedback\n[[FEEDBACK]]\n\nRevise your output to generate a faithful argument reconstruction.'
            input_ = input_.replace('[[PREMISES]]', '\n'.join(sections_final['valid_premises'])).replace('[[CONCLUSION]]', sections_final['valid_conclusion']).replace('[[FEEDBACK]]', feedback_faithfulness)
            if formal_fallacy_flag:
                input_ += f'\n\n{RECON_FORMAT_FORMAL_FALLACY}'
            else:
                input_ += f'\n\n{RECON_FORMAT}'
        else: # validity f/b
            input_ = 'The formalized premises do not necessarily lead to the conclusion. Revise your output.'
            input_ += f'\n\n{RECON_FORMAT}'
        message_list.append(input_)

        attempt = 0
        while attempt < self.max_attempts:
            output_, input_tokens_, output_tokens_ = self.api_call(message_list, temperature=1.0)
            
            print("<| Do Reconstruction |>")
            print(output_)

            # Move Back to original pattern - New post-process rather makes an error
            if formal_fallacy_flag:
                pattern = r"## Premises\n(.*?)" \
                r"## Intermediate Conclusions\n(.*?)" \
                r"## Conclusion\n(.*?)" \
                r"# Formalized Argument\n\n## Defined Variables/Predicates\n(.*?)" \
                r"## Formalized Premises\n(.*?)" \
                r"## Formalized Intermediate Conclusions\n(.*?)" \
                r"## Formalized Conclusion\n(.*?)\Z"
            else:
                pattern = r"## Premises\n(.*?)" \
                r"## Intermediate Conclusions\n(.*?)" \
                r"## Conclusion\n(.*?)" \
                r"## Logical Connections\n(.*?)" \
                r"# Formalized Argument\n\n## Defined Variables/Predicates\n(.*?)" \
                r"## Formalized Premises\n(.*?)" \
                r"## Formalized Intermediate Conclusions\n(.*?)" \
                r"## Formalized Conclusion\n(.*?)" \
                r"## Deductive Proof\n(.*?)\Z"
                
            match = re.search(pattern, output_, re.DOTALL)
            
            if match:
                message_list.append(output_)
                if formal_fallacy_flag:
                    # Formal-fallacy prompt omits "Logical Connections" and
                    # "Deductive Proof" sections (7 capture groups).
                    sections = {
                        "premises": match.group(1).strip(),
                        "intermediate_conclusions": match.group(2).strip(),
                        "conclusion": match.group(3).strip(),
                        "connections": "",
                        "definition": match.group(4).strip(),
                        "formalized_premises": match.group(5).strip(),
                        "formalized_intermediate_conclusions": match.group(6).strip(),
                        "formalized_conclusion": match.group(7).strip(),
                        "proof": ""
                    }
                else:
                    sections = {
                        "premises": match.group(1).strip(),
                        "intermediate_conclusions": match.group(2).strip(),
                        "conclusion": match.group(3).strip(),
                        "connections": match.group(4).strip(),
                        "definition": match.group(5).strip(),
                        "formalized_premises": match.group(6).strip(),
                        "formalized_intermediate_conclusions": match.group(7).strip(),
                        "formalized_conclusion": match.group(8).strip(),
                        "proof": match.group(9).strip()
                    }
                return sections, message_list, input_tokens_, output_tokens_
            
            attempt += 1
        
        raise ValueError(f'raw output:\n{output_}')
    
    def execute_program(self, code):
        filename = join(self.cache_dir, f'tmp.py')
        with open(filename, "w") as f:
            f.write(code)
        try:
            output = check_output(["python3", filename], stderr=subprocess.STDOUT, timeout=300.0) # Increase timeout limit if needed
        except subprocess.CalledProcessError as e:
            outputs = e.output.decode("utf-8").strip() # whole error message, instead of the last line
            return (None, None), outputs
        except subprocess.TimeoutExpired:
            return (None, None), 'TimeoutError'
        result = output.decode("utf-8").strip().splitlines()
        if len(result) != 2:
            err_msg = f'An executed output of the python program is:\n\n{output}\n\nThe python program should only print two things as desribed in the previous chat.'
            return (None, None), err_msg
        
        return result, ""
    
    def debug_program(self, error, message_list):
        input_ = self.prompt_debug.replace('[[ERROR]]', error)
        message_list.append(input_)

        attempt = 0
        while attempt < self.max_attempts:
            output_, input_tokens_, output_tokens_ = self.api_call(message_list, temperature=1.0)

            # Move Back to original pattern - New post-process rather makes an error
            pattern = r"### Reasoning\n(.*?)" \
            r"### Revised Python Program\n(.*?)\Z"

            match = re.search(pattern, output_, re.DOTALL)

            if match:
                message_list.append(output_)
                raw_program = match.group(2).strip()
                z3_program = raw_program.split('```python')[-1].split('```')[0].strip()    # remove ```python at first, and ``` at last
                return z3_program, message_list, input_tokens_, output_tokens_
            
            attempt += 1
        
        raise ValueError(f'raw output:\n{output_}')
    
    def is_valid(self, sections):
        definition = sections['definition']
        formalized_premises = sections['formalized_premises']
        formalized_conclusion = sections['formalized_conclusion']
        proof = sections['proof']

        input_ = self.prompt_validity.replace('[[DEFINITION]]', definition).replace('[[PREMISES]]', formalized_premises).replace('[[CONCLUSION]]', formalized_conclusion).replace('[[PROOF]]', proof)

        attempt = 0
        while attempt < self.max_attempts:
            output_, input_tokens_, output_tokens_ = self.api_call(input_, temperature=1.0)
            print("<| Check Validity |>")
            print(output_)

            # Move Back to original pattern - New post-process rather makes an error
            pattern = r"### Necessary Formalized Premises\n(.*?)" \
            r"### Python Program\n(.*?)" \
            r"### Final Formalized Conclusion\n(.*?)\Z"
            
            match = re.search(pattern, output_, re.DOTALL)
            
            if match:
                sections_validity = {
                    "necessary_formalized_premises": match.group(1).strip(),
                    "final_formalized_conclusion": match.group(3).strip(),
                    "z3_program": match.group(2).strip()
                }
                break
            
            attempt += 1
        
        if attempt >= self.max_attempts:
            raise ValueError(f'raw output:\n{output_}')

        z3_program = sections_validity['z3_program'].split('```python')[-1].split('```')[0].strip()   # remove ```python at first, and ``` at last
        (output, list_valid_premises), error_message = self.execute_program(z3_program)

        cnt = 1
        message_list = []
        message_list.append(input_)
        message_list.append(output_)
        while output is None:
            print(error_message)

            if cnt >= self.max_num_debug:
                print('failed to debug the program')
                break

            target_program, message_list, input_tokens_for_debug, output_tokens_for_debug = self.debug_program(error_message, message_list)
            input_tokens_ += input_tokens_for_debug
            output_tokens_ += output_tokens_for_debug
            (output, list_valid_premises), error_message = self.execute_program(target_program)

            cnt += 1
        
        if cnt > 1:
            sections_validity['z3_program'] = f'```python\n{target_program}\n```'
        
        # choose a smallest subset of premises
        python_dict = sections_validity['necessary_formalized_premises'].split('```python')[-1].split('```')[0].strip() # remove ```python at first, and ``` at last
        try:
            premises_dict = eval(python_dict)
        except:
            print(f'raw python dict:\n{python_dict}')

            start = python_dict.find('{')
            end   = python_dict.rfind('}')
            python_dict = python_dict[start + 1 : end]

            premises_dict = dict()
            current_key = None
            current_value_lines = []
            
            for line in python_dict.splitlines():
                stripped = line.strip()
                if not stripped:
                    continue
                
                # Check if this line starts a new key-value pair
                # A new key starts with a quote (either " or ') followed by text and a colon
                if ':' in stripped and (stripped.startswith('"') or stripped.startswith("'")):
                    # Save the previous key-value pair if exists
                    if current_key is not None:
                        full_value = ' '.join(current_value_lines).rstrip(',')
                        premises_dict[current_key] = full_value
                    
                    # Parse the new key-value pair
                    key, value = stripped.split(':', 1)
                    current_key = key.strip().strip('\'"')
                    current_value_lines = [value.strip()]
                else:
                    # This is a continuation line for the current key
                    if current_key is not None:
                        current_value_lines.append(stripped)
            
            # Don't forget the last key-value pair
            if current_key is not None:
                full_value = ' '.join(current_value_lines).rstrip(',')
                premises_dict[current_key] = full_value
        
        valid_premises_list = []
        for key, value in premises_dict.items():
            if output == 'valid':
                if key in eval(list_valid_premises):
                    valid_premises_list.append(f'{key}: {value}')
            else:
                valid_premises_list.append(f'{key}: {value}')
        
        sections_validity['validity'] = output
        sections_validity['valid_formalized_premises'] = valid_premises_list
        if output == 'valid':
            return True, sections_validity, '\n'.join(valid_premises_list), input_tokens_, output_tokens_
        elif output == 'invalid':
            return False, sections_validity, '\n'.join(valid_premises_list), input_tokens_, output_tokens_
        else:
            return None, sections_validity, '\n'.join(valid_premises_list), input_tokens_, output_tokens_
    
    def is_faithful(self, topic, background, argument, sections_final, formal_fallacy_flag, informal_fallacy_flag):
        if formal_fallacy_flag and informal_fallacy_flag:
            arg_type = "both formally fallacious and informally fallacious"
        elif formal_fallacy_flag:
            arg_type = "formally fallacious"
        elif informal_fallacy_flag:
            arg_type = "informally fallacious"
        else:
            arg_type = "non-fallacious"

        input_ = self.prompt_faithfulness.replace('[[Topic]]', topic).replace('[[Background]]', background).replace('[[ARG]]', argument).replace('[[PREMISES]]', '\n'.join(sections_final['valid_premises'])).replace('[[CONCLUSION]]', sections_final['valid_conclusion']).replace('[[ARG_TYPE]]', arg_type)

        attempt = 0
        while attempt < self.max_attempts:
            output_, input_tokens_, output_tokens_ = self.api_call(input_, temperature=0.0) # LLM-as-a-judge style -- low temperature
            print("<| Check Faithfulness |>")
            print(output_)

            # Move Back to original pattern - New post-process rather makes an error
            pattern = r"# Reasoning\n(.*?)" \
            r"# Faithfulness\n(.*?)\Z"
            
            match = re.search(pattern, output_, re.DOTALL)
            
            if match:
                feedback = match.group(1).strip()
                if 'yes' in match.group(2).strip().lower():
                    faithfulness = True
                elif 'no' in match.group(2).strip().lower():
                    faithfulness = False
                else:
                    attempt += 1
                    continue
                return faithfulness, feedback, input_tokens_, output_tokens_
            
            attempt += 1
        
        raise ValueError(f'raw output:\n{output_}')

    def generate_valid_reconstruction(self, sections, valid_premises, valid_conclusion):
        definition = sections['definition']
        necessary_formalized_premises = valid_premises
        formalized_conclusion = valid_conclusion

        input_ = self.prompt_deformalization.replace('[[DEFINITION]]', definition).replace('[[PREMISES]]', necessary_formalized_premises).replace('[[CONCLUSION]]', formalized_conclusion)

        attempt = 0
        while attempt < self.max_attempts:
            output_, input_tokens_, output_tokens_ = self.api_call(input_, temperature=1.0)
            print("<| Do Deformalization |>")
            print(output_)

            # Move Back to original pattern - New post-process rather makes an error
            pattern = r"### NL Premises\n(.*?)" \
            r"### NL Conclusion\n(.*?)\Z"

            match = re.search(pattern, output_, re.DOTALL)

            if match:
                tmp = match.group(1).strip()
                linesp = '\n'
                if '\n\n' in tmp:
                    linesp = '\n\n'
                
                sections_final = {
                    "valid_premises": tmp.split(linesp),
                    "valid_conclusion": match.group(2).strip(),
                }
                return sections_final, input_tokens_, output_tokens_
            
            attempt += 1
        
        raise ValueError(f'raw output:\n{output_}')
    
    def generate(self):
        # load data
        dataset = self.load_raw_dataset()

        ind = 0
        for data in tqdm(dataset, total=len(dataset)):
            
            # If you want to skip first N data
            # N = 1
            # if ind < N:
            #     ind += 1
            #     continue

            argument = data['argument']
            topic = data['title']
            background = data['background']

            total_input_tokens = 0
            total_output_tokens = 0

            # step 0: Fallacy Detection
            formal_fallacy_flag = False
            informal_fallacy_flag = False

            is_formal_fallacy, is_informal_fallacy, sections_fallacy_detection, input_tokens_, output_tokens_ = self.detect_fallacy(topic, background, argument)
            total_input_tokens += input_tokens_
            total_output_tokens += output_tokens_

            if is_formal_fallacy:
                formal_fallacy_flag = True
            if is_informal_fallacy:
                informal_fallacy_flag = True

            cnt = 1
            message_list = []
            validation_list = []
            feedback_faithfulness = None
            sections_final, sections_validity = None, None
            while True:
                print(f'Trial #{cnt}')
                # step 1: Reconstruct Argument
                sections, message_list, input_tokens_, output_tokens_ = self.generate_reconstruction(formal_fallacy_flag, informal_fallacy_flag, sections_fallacy_detection, topic, background, argument, message_list, feedback_faithfulness, sections_final)
                total_input_tokens += input_tokens_
                total_output_tokens += output_tokens_

                # step 2: Check Validity by SAT Solver
                # If formal fallacy, do NOT run validity check
                if not formal_fallacy_flag:
                    validity, sections_validity, valid_premises, input_tokens_, output_tokens_ = self.is_valid(sections)
                    total_input_tokens += input_tokens_
                    total_output_tokens += output_tokens_
                    validation_list.append(sections_validity)
                    print(f'validity: {validity}')

                    # If this round's validity check failed, clear any stale
                    # faithfulness-feedback state so the NEXT iteration uses the
                    # validity-feedback branch (not a leftover faithfulness one).
                    if not validity:
                        feedback_faithfulness = None
                        sections_final = None
                else:
                    validation_list.append(None)
                    print('Formal Fallacy! Validity: False (By Definition)')

                if formal_fallacy_flag or validity or cnt >= self.max_num_recon:
                    # step 3: Translate Back from FOL to NL
                    if formal_fallacy_flag:
                        sections_final, input_tokens_, output_tokens_ = self.generate_valid_reconstruction(sections, sections['formalized_premises'], sections['formalized_conclusion'])
                        total_input_tokens += input_tokens_
                        total_output_tokens += output_tokens_
                    else:
                        sections_final, input_tokens_, output_tokens_ = self.generate_valid_reconstruction(sections, valid_premises, sections_validity['final_formalized_conclusion'])
                        total_input_tokens += input_tokens_
                        total_output_tokens += output_tokens_
                    
                    # step 4: Check Faithfulness by LLM-as-a-judge
                    faithfulness, feedback_faithfulness, input_tokens_, output_tokens_ = self.is_faithful(topic, background, argument, sections_final, formal_fallacy_flag, informal_fallacy_flag)
                    total_input_tokens += input_tokens_
                    total_output_tokens += output_tokens_
                    
                    print(f'faithfulness: {faithfulness}')

                    if faithfulness or cnt >= self.max_num_recon:
                        break

                cnt += 1
                time.sleep(5)                  # respect rate limits
            
            # database for argument reconstruction
            sections_summary = dict()
            sections_summary['fallacy_detection'] = sections_fallacy_detection
            sections_summary['reconstruction'] = sections
            sections_summary['check_validity'] = sections_validity
            sections_summary['streamlined'] = sections_final
            sections_summary['check_faithfulness'] = {
                'faithfulness': faithfulness,
                'feedback_faithfulness': feedback_faithfulness
            }
            
            pricing_info = {
                'total_input_tokens': total_input_tokens,
                'total_output_tokens': total_output_tokens,
                'total_price': self.count_price(total_input_tokens, total_output_tokens)
            }

            final_output_dict = {
                'title': topic,
                'background': background,
                'argument': argument,
                'sections': sections_summary,
                'messages': message_list,
                'validation_messages': validation_list,
                'pricing_info': pricing_info
            }

            # save the output file (for every paper)
            filename_output = os.path.join(self.save_path, f'reconstruction_{self.subset}_{self.model}.json') ###
            if not os.path.exists(filename_output):
                with open(filename_output, 'w', encoding='utf-8') as f:
                    json.dump([], f, ensure_ascii=False, indent=2)
            
            with open(filename_output, 'r+', encoding='utf-8') as f:
                existed_data = json.load(f)
                existed_data.append(final_output_dict)
                f.seek(0)
                json.dump(existed_data, f, ensure_ascii=False, indent=2)
            
            ind += 1

            # if ind == 100:
            #     break

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data/Sample')
    parser.add_argument('--data_filename', type=str, default='sample.json')
    parser.add_argument('--use_general_reconstruction', type=bool, default=True)
    parser.add_argument('--use_specific_reconstruction', type=bool, default=False)
    parser.add_argument('--save_path', type=str, default='./output')
    parser.add_argument('--prompt_path', type=str, default='./prompts/GAAR')
    parser.add_argument('--subset', type=str, default='sample')
    parser.add_argument('--model_name', type=str, default='claude-sonnet-4-5-20250929')
    parser.add_argument('--max_num_recon', type=int, default=10) ### 5
    parser.add_argument('--max_num_debug', type=int, default=5)
    parser.add_argument('--max_attempts', type=int, default=5)
    args = parser.parse_args()

    # Check if the arguments are valid
    if args.use_general_reconstruction and args.use_specific_reconstruction:
        raise ValueError('You cannot use both general and specific reconstruction at the same time.')
    if not args.use_general_reconstruction and not args.use_specific_reconstruction:
        raise ValueError('You must use either general or specific reconstruction.')
    return args

if __name__ == '__main__':
    args = parse_args()
    print(args)

    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    
    streamliner = LogicalStreamliner(args)
    streamliner.generate()
