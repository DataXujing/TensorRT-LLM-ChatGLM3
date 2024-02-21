# 编写Gradio调用函数
# import mdtex2html
from service.config import LangChainCFG
# from service.configuration_chatglm import ChatGLMConfig

# from langchain_chatglm3 import *
from langchain_chatglm3_triton import *


import gradio as gr


#将文本中的字符转为网页上可以支持的字符，避免被误认为是HTML标签
def parse_text(text):
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f'<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", "\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>"+line
    text = "".join(lines)
    return text

#采用流聊天方式（stream_chat）调用模型，使得生成答案有逐字生成的效果
#chatglm3实现了stream_chat接口
# def predict(input, chatbot, max_length, top_p, temperature, history, past_key_values):
#     chatbot.append((parse_text(input), parse_text(input)))
#     for response, history, past_key_values in model.stream_chat(tokenizer, input, history, past_key_values=past_key_values,
#                                                                 return_past_key_values=True,
#                                                                 max_length=max_length, top_p=top_p,
#                                                                 temperature=temperature):
#         chatbot[-1] = (parse_text(input), parse_text(response)) # 一直在替换response(repose每次都多一个token)

#         yield chatbot, history, past_key_values

# 在这里定义 application
config = LangChainCFG()
application = LangChainApplication(config)

def predict(input_text, chatbot, max_length, top_p, temperature, history, past_key_values):
    application.knowledge_service.init_knowledge_base()
    chatbot.append((parse_text(input_text), parse_text(input_text)))
    # chat_glm_model = ChatGLMForConditionalGeneration(config=ChatGLMConfig)
    
    response_dict = application.get_knowledeg_based_answer(parse_text(input_text), history_len=5, temperature=0.1, top_p=0.9, top_k=4, chat_history=history)
    if 'result' in response_dict and isinstance(response_dict['result'], str):
        result_text = response_dict['result']
        response = parse_text(result_text)
        chatbot[-1] = (parse_text(input_text), response)
        yield chatbot, history, past_key_values
    else:
        # Handle the case where 'result' key is missing or not a string
        # You can modify this part based on your requirements
        yield chatbot, history, past_key_values
    
    # for response, history, past_key_values in chat_glm_model.stream_chat(tokenizer, input, history, past_key_values=past_key_values,
    #                                                             return_past_key_values=True,
    #                                                             max_length=max_length, top_p=top_p,
    #                                                             temperature=temperature):
        
        

#去除输入框的内容
def reset_user_input():
    return gr.update(value='')

#清除状态
def reset_state():
    return [], [], None


#运行Gradio界面，运行成功后点击“Running on public URL”后的网页链接即可体验

with gr.Blocks() as demo:
    gr.HTML("""<h1 align="center">基于LLM和知识库的RAG结构化诊断报告生成</h1>""")

    chatbot = gr.Chatbot()  # gr的聊天界面
    with gr.Row():
        with gr.Column(scale=4):
            with gr.Column(scale=12):
                user_input = gr.Textbox(show_label=False, placeholder="Input...", lines=10,container=False)
            with gr.Column(min_width=32, scale=1):
                submitBtn = gr.Button("Submit", variant="primary")
        with gr.Column(scale=1):
            emptyBtn = gr.Button("Clear History")
            max_length = gr.Slider(0, 32768, value=8192, step=1.0, label="Maximum length", interactive=True)
            top_p = gr.Slider(0, 1, value=0.8, step=0.01, label="Top P", interactive=True)
            temperature = gr.Slider(0, 1, value=0.95, step=0.01, label="Temperature", interactive=True)

    history = gr.State([])  # State
    past_key_values = gr.State(None)

    submitBtn.click(predict, [user_input, chatbot, max_length, top_p, temperature, history, past_key_values],
                    [chatbot, history, past_key_values], show_progress=True)
    submitBtn.click(reset_user_input, [], [user_input])

    emptyBtn.click(reset_state, outputs=[chatbot, history, past_key_values], show_progress=True)



demo.queue().launch(share=True, inbrowser=True,server_name="0.0.0.0", server_port=8005)