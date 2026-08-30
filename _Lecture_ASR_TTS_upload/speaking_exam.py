import base64
from pathlib import Path

import pandas as pd
import streamlit as st
from audio_recorder_streamlit import audio_recorder
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_openai import ChatOpenAI
from openai import OpenAI
from pydantic import BaseModel, Field


# Set page configuration for wide layout
st.set_page_config(layout="wide")

if "curr_page" not in st.session_state:
    st.session_state["curr_page"] = "home"
    st.session_state["curr_topic"] = "home"

if "prev_audio_bytes" not in st.session_state:
    st.session_state.prev_audio_bytes = None

if "exam_context" not in st.session_state:
    st.session_state.exam_context = {}


client = OpenAI()
APP_DIR = Path(__file__).resolve().parent


WRITING_EXAM_CONFIGS = {
    "writing__responding_to_an_email": {
        "context_heading": "받은 이메일",
        "answer_label": "영어 답장",
        "answer_placeholder": "이메일의 요청 사항에 맞춰 영어로 답장하세요.",
        "generation_prompt": (
            "영어 Writing 시험의 '이메일 답장하기' 문제로 사용할 가상의 영어 이메일을 "
            "하나 작성하라. 보낸 사람, 받는 사람, 제목을 포함하고 응시자가 답장에서 "
            "반드시 다뤄야 할 요청을 2~3개 넣어라. 120~180단어로 작성하고 문제 외의 "
            "설명이나 모범 답안은 출력하지 마라."
        ),
        "evaluation_instruction": (
            "사용자의 영어 답장이 이메일의 목적과 요청 사항에 적절히 응답했는지 평가하라. "
            "내용의 완전성, 어조, 구성, 문법과 어휘를 함께 고려하라."
        ),
    },
    "writing__summarization": {
        "context_heading": "요약할 제시문",
        "answer_label": "영어 요약문",
        "answer_placeholder": "핵심 내용을 2~4문장으로 요약하세요.",
        "generation_prompt": (
            "영어 Writing 시험의 '제시문 요약하기' 문제로 사용할 가상의 영어 단락을 "
            "하나 작성하라. 중심 생각과 이를 뒷받침하는 세부 내용이 분명해야 하며 "
            "180~240단어로 작성하라. 제시문 외의 설명이나 요약문은 출력하지 마라."
        ),
        "evaluation_instruction": (
            "사용자의 영어 요약문이 제시문의 중심 생각과 핵심 세부 내용을 정확하고 간결하게 "
            "담았는지 평가하라. 불필요한 정보, 원문에 없는 주장, 구성, 문법과 어휘를 고려하라."
        ),
    },
    "writing__writing_opinion": {
        "context_heading": "의견 쓰기 주제",
        "answer_label": "영어 의견문",
        "answer_placeholder": "입장을 밝히고 이유와 예시를 들어 영어로 작성하세요.",
        "generation_prompt": (
            "영어 Writing 시험의 '자신의 의견 쓰기' 문제로 사용할 논쟁적인 일상·교육·기술 "
            "주제 하나를 영어로 제시하라. 상반된 두 관점을 간단히 소개하고 응시자의 의견과 "
            "근거를 묻는 하나의 문제로 작성하라. 문제 외의 설명이나 모범 답안은 출력하지 마라."
        ),
        "evaluation_instruction": (
            "사용자의 영어 의견문이 분명한 입장과 타당한 이유 및 예시를 제시했는지 평가하라. "
            "논리적 구성, 설득력, 문단 연결, 문법과 어휘를 함께 고려하라."
        ),
    },
}


def generate_writing_context(prompt: str) -> str:
    model = ChatOpenAI(model="gpt-5.6-luna", http_socket_options=())
    return model.invoke(prompt).content


def evaluate_writing_answer(instruction: str, context: str, user_answer: str):
    class Evaluation(BaseModel):
        reason: str = Field(description="점수의 근거를 한국어로 설명")
        feedback: str = Field(description="답안을 개선하기 위한 구체적인 피드백을 한국어로 작성")
        score: int = Field(description="Writing 답안 점수. 0~10점", ge=0, le=10)

    parser = JsonOutputParser(pydantic_object=Evaluation)
    format_instructions = parser.get_format_instructions()
    human_prompt_template = HumanMessagePromptTemplate.from_template(
        "# 평가 지침\n{instruction}\n\n"
        "# 시험 제시문\n{context}\n\n"
        "# 사용자 답안\n{input}\n\n"
        "{format_instructions}",
        partial_variables={"format_instructions": format_instructions},
    )
    prompt = ChatPromptTemplate.from_messages([human_prompt_template])
    model = ChatOpenAI(model="gpt-5.6-luna", http_socket_options=())
    return (prompt | model | parser).invoke(
        {"instruction": instruction, "context": context, "input": user_answer}
    )


def autoplay_audio(file_path: str):
    with open(file_path, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        md = f"""
            <audio controls autoplay="true">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            """
        st.markdown(
            md,
            unsafe_allow_html=True,
        )

def recognize_speech():
    user_input = ""
    # 질문에 답하기
    audio_bytes = audio_recorder("talk", pause_threshold=3.0,)
    if audio_bytes == st.session_state.prev_audio_bytes:
        audio_bytes = None
    st.session_state.prev_audio_bytes = audio_bytes

    try:
        if audio_bytes:
            with st.spinner("음성 인식중..."):
                with open("./tmp_audio.wav", "wb") as f:
                    f.write(audio_bytes)

                with open("./tmp_audio.wav", "rb") as f: 
                    transcript = client.audio.transcriptions.create(
                        model="whisper-1",
                        file=f,
                        language="en"
                    )
                    user_input = transcript.text
    except Exception as e:
        print(e)
        pass
    return user_input


# Assuming you have a dictionary that holds your data like below:
speaking_topic_to_topic_info_map = {
    'speaking__listen_and_answer': {'display_name': '듣고 질문에 답하기', 'emoji': '💭'},
    'speaking__express_an_opinion': {'display_name': '의견 말하기', 'emoji': '🗣️'},
    'speaking__debate': {'display_name': '토론하기', 'emoji': '👩‍'},
    'speaking__describe_img': {'display_name': '사진 묘사하기', 'emoji': '🏞️'},
    'speaking__describe_charts': {'display_name': '도표 보고 발표하기', 'emoji': '📊'},
}

writing_topic_to_topic_info_map = {
    'writing__dictation': {'display_name': '받아쓰기 시험 유형 만들기', 'emoji': '✏️'},
    'writing__responding_to_an_email': {'display_name': '이메일 답장하기', 'emoji': '✉️'},
    'writing__summarization': {'display_name': '제시문 내용을 요약하기', 'emoji': '✍️'},
    'writing__writing_opinion': {'display_name': '자신의 의견쓰기', 'emoji': '📝'},
}


# def go_to_topic(topic):
#     st.session_state["curr_page"] = topic
#     st.session_state["curr_topic"] = topic

def go_to_topic(topic):
    st.session_state["curr_page"] = topic
    st.session_state["curr_topic"] = topic
    st.session_state["exam_context"] = {}
    st.session_state["prev_audio_bytes"] = None

def go_to_result():
    st.session_state["curr_page"] = "result"

# Create a function to display each topic in the grid
def display_topic(topic, topic_info, key):
    with st.container(border=True):
        st.write(f"{topic_info['emoji']} **{topic_info['display_name']}**")
        st.button("시작", key=f"start_{topic}_{key}", on_click=go_to_topic, kwargs=dict(topic=topic))


con = st.container()
if st.session_state["curr_page"] == "home":
    with con:
        st.title("Speaking & Writing 어학 시험")


        tab1, tab2 = st.tabs(["Speaking 시험", "Writing 시험"])

        with tab1:

            cols = st.columns(2)
            for i, (topic, topic_info) in enumerate(speaking_topic_to_topic_info_map.items()):
                with cols[i % 2]:  # This will alternate between the two columns
                    display_topic(topic, topic_info, i)
        
        with tab2:
            cols = st.columns(2)
            for i, (topic, topic_info) in enumerate(writing_topic_to_topic_info_map.items()):
                with cols[i % 2]:  # This will alternate between the two columns
                    display_topic(topic, topic_info, i)


elif st.session_state["curr_page"] == "speaking__listen_and_answer":
    topic_info = speaking_topic_to_topic_info_map[st.session_state.curr_topic]
    st.title(topic_info['display_name'])

    # random 하게 질문 하나 가져오기
    @st.cache_data
    def load_listen_and_answer_data():
        df = pd.read_csv("./data/speaking__listen_and_answer/question_and_audio.csv")
        return df

    df = load_listen_and_answer_data()

    if "question" not in st.session_state.exam_context:
        sample = df.sample(n=1).iloc[0]

        question = sample["question"]
        audio_file_path = sample["audio_file_path"]

        st.session_state.exam_context["sample"] = sample
        st.session_state.exam_context["question"] = question
        st.session_state.exam_context["audio_file_path"] = audio_file_path


    if st.button("시험 시작"):
        st.session_state.exam_context["exam_start"] = True
        st.session_state.exam_context["do_speech"] = True

    if st.session_state.exam_context.get("exam_start", False):
        if st.session_state.exam_context["do_speech"]:
            autoplay_audio(st.session_state.exam_context["audio_file_path"])
            st.session_state.exam_context["do_speech"] = False

        if not st.session_state.exam_context["do_speech"]:
            # recognized_text = recognize_speech()
            # st.session_state.exam_context["user_answer"] = recognized_text

            recognized_text = recognize_speech()
            if recognized_text:
                st.session_state.exam_context["user_answer"] = recognized_text

        if st.session_state.exam_context.get("user_answer"):

            with st.container(border=True):
                answer_text = f"""
                - Question: {st.session_state.exam_context["question"]}
                - Your Answer: {st.session_state.exam_context.get("user_answer")}
                """

                st.markdown(answer_text)
                

            def get_speaking__listen_and_answer_result(answer_text):
                model = ChatOpenAI(model="gpt-5.6-luna", http_socket_options=())
                class Score(BaseModel):
                    reason: str = Field(description="Question에 대해 Your Answer가 적절한지에 대해 추론하라. 한국어로.")
                    score: int = Field(description="Question에 대해 Your Answer가 적절한지에 대해 0~10점 사이의 점수를 부여하라")
                parser = JsonOutputParser(pydantic_object=Score)
                format_instruction = parser.get_format_instructions()

                human_msg_prompt_template = HumanMessagePromptTemplate.from_template(
                    "{input}\n---\nQuestion에 대해 Your Answer가 적절한지에 대해 추론해서 0~10점 사이의 점수를 부여해라. 다음의 포맷에 맞춰 응답해라.  : {format_instruction}",
                    partial_variables={"format_instruction": format_instruction})

                prompt_template = ChatPromptTemplate.from_messages(
                    [
                        human_msg_prompt_template
                    ],
                )
                
                chain = prompt_template | model | parser
                return chain.invoke({"input": answer_text})

                
            with st.container(border=True):
                """
                ### 평가 결과
                """

                with st.spinner("채점중..."):
                    result = get_speaking__listen_and_answer_result(answer_text)

                f"""
                {result['reason']}

                #### 총점: {result['score']}

                """

####################################
# 
elif st.session_state["curr_page"] == "speaking__express_an_opinion":
    topic_info = speaking_topic_to_topic_info_map[st.session_state.curr_topic]
    st.title(topic_info['display_name'])

    # random 하게 질문 하나 가져오기
    @st.cache_data
    def load_speaking__express_an_opinion_data():
        df = pd.read_csv("./data/speaking__express_an_opinion/question_and_audio.csv")
        return df

    df = load_speaking__express_an_opinion_data()

    if "question" not in st.session_state.exam_context:
        sample = df.sample(n=1).iloc[0]

        question = sample["question"]
        audio_file_path = sample["audio_file_path"]

        st.session_state.exam_context["sample"] = sample
        st.session_state.exam_context["question"] = question
        st.session_state.exam_context["audio_file_path"] = audio_file_path


    if st.button("시험 시작"):
        st.session_state.exam_context["exam_start"] = True
        st.session_state.exam_context["do_speech"] = True

    if st.session_state.exam_context.get("exam_start", False):
        if st.session_state.exam_context["do_speech"]:
            autoplay_audio(st.session_state.exam_context["audio_file_path"])
            st.session_state.exam_context["do_speech"] = False

        if not st.session_state.exam_context["do_speech"]:
            # recognized_text = recognize_speech()
            # st.session_state.exam_context["user_answer"] = recognized_text

            recognized_text = recognize_speech()
            if recognized_text:
                st.session_state.exam_context["user_answer"] = recognized_text

        if st.session_state.exam_context.get("user_answer"):

            with st.container(border=True):
                answer_text = f"""
                - Question: {st.session_state.exam_context["question"]}
                - Your Answer: {st.session_state.exam_context.get("user_answer")}
                """

                st.markdown(answer_text)
                
            with st.container(border=True):
                def get_speaking__express_opinion_result(answer_text):
                    model = ChatOpenAI(model="gpt-5.6-luna", http_socket_options=())
                    class Score(BaseModel):
                        reason: str = Field(description="Question에 대해 의견을 말하는 시험이다. 의견을 적절히 구조적으로 응답했는지 추론하라. 한국어로.")
                        score: int = Field(description="Question에 대해 Your Answer가 충분히 논리적으로 의견을 표현했는지에 대해 0~10점 사이의 점수를 부여하라.")
                    parser = JsonOutputParser(pydantic_object=Score)
                    format_instruction = parser.get_format_instructions()

                    human_msg_prompt_template = HumanMessagePromptTemplate.from_template(
                        "{input}\n---\nQuestion에 대해 Your Answer가 충분히 논리적으로 의견을 표현했는지에 대해 0~10점 사이의 점수를 부여하라. 다음의 포맷에 맞춰 응답해라.  : {format_instruction}",
                        partial_variables={"format_instruction": format_instruction})

                    prompt_template = ChatPromptTemplate.from_messages(
                        [
                            human_msg_prompt_template
                        ],
                    )
                    
                    chain = prompt_template | model | parser
                    return chain.invoke({"input": answer_text})

                """
                ### 평가 결과
                """

                with st.spinner("채점중..."):
                    result = get_speaking__express_opinion_result(answer_text)

                f"""
                {result['reason']}

                #### 총점: {result['score']}

                """


elif st.session_state["curr_page"] == "speaking__debate":

    st.title("토론하기")

    con1 = st.container()
    con2 = st.container()

    user_input = ""

    if "model" not in st.session_state.exam_context:
        st.session_state.exam_context["model"] = ChatOpenAI(
            model="gpt-5.6-luna", http_socket_options=()
        )

    if "messages" not in st.session_state.exam_context:
        system_prompt = """\
- 너는 AI 시험 감독이다.
- user의 영어 실력을 위해 어떠한 주제에 대해 서로 질문과 답을하며 토론한다."""

        model = st.session_state.exam_context["model"]
        question = model.invoke("Create a controversial question for me.").content

        st.session_state.exam_context["messages"] = [SystemMessage(content=system_prompt),
                                                     AIMessage(content=question),
                                                     ]

        speech_file_path = "tmp_speak.mp3"
        response = client.audio.speech.create(
            model="tts-1",
            voice="alloy", # alloy, echo, fable, onyx, nova, and shimmer
            input=question
        )
        response.stream_to_file(speech_file_path)
        autoplay_audio(speech_file_path)

    with con1:
        for message in st.session_state.exam_context['messages']:
            if isinstance(message, SystemMessage):
                continue
            role = 'user' if message.type == 'human' else 'assistant'
            with st.chat_message(role):
                st.markdown(message.content)

    with con2:
        user_input = recognize_speech()

    with con1:
    
        turn_len = len(st.session_state.exam_context['messages'])
        max_turn_len = 5

        if user_input and turn_len < max_turn_len:
            st.session_state.exam_context['messages'].append(HumanMessage(content=user_input))

            with st.chat_message("user"):
                st.markdown(user_input)
            
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""

                model = st.session_state.exam_context["model"]

                for chunk in model.stream(st.session_state.exam_context['messages']):
                    full_response += (chunk.content or "")
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)

                speech_file_path = "tmp_speak.mp3"
                response = client.audio.speech.create(
                model="tts-1",
                voice="alloy", # alloy, echo, fable, onyx, nova, and shimmer
                input=full_response
                )
                response.stream_to_file(speech_file_path)

                autoplay_audio(speech_file_path)

            st.session_state.exam_context['messages'].append(AIMessage(content=full_response))

        if turn_len >= max_turn_len:

            def get_speaking__debate_result(conversation):
                model = ChatOpenAI(model="gpt-5.6-luna", http_socket_options=())
                class Score(BaseModel):
                    reason: str = Field(description="주어진 대화에 대해 User가 얼마나 논리적이고 유창하게 영어로 응답하였는지 추론하라. 한국어로.")
                    score: int = Field(description="주어진 대화에서 User의 응답에 대해 유창성과 논리성을 고려하여 0~10점 사이의 점수를 부여하라.")
                parser = JsonOutputParser(pydantic_object=Score)
                format_instruction = parser.get_format_instructions()

                human_msg_prompt_template = HumanMessagePromptTemplate.from_template(
                    "{input}\n---\n주어진 대화에서 User의 응답에 대해 유창성과 논리성을 고려하여 0~10점 사이의 점수를 부여하라. 다음의 포맷에 맞춰 응답해라.  : {format_instruction}",
                    partial_variables={"format_instruction": format_instruction})

                prompt_template = ChatPromptTemplate.from_messages(
                    [
                        human_msg_prompt_template
                    ],
                )
                
                chain = prompt_template | model | parser
                return chain.invoke({"input": conversation})

                
            with st.container(border=True):
                """
                ### 평가 결과
                """

                with st.spinner("채점중..."):

                    conversation = ""
                    for msg in st.session_state.exam_context["messages"]:
                        role = 'User' if msg.type == 'human' else 'AI'
                        conversation += f"{role}: {msg.content}"

                    result = get_speaking__debate_result(conversation)

                grade = ""

                if result['score'] >= 8:
                    grade = "Advanced"
                elif 4 < result['score'] < 8:
                    grade = "Intermediate"
                elif result['score'] <= 4:
                    grade = "Novice"

                grade = f"{grade}, {result['score']}"

                f"""
                {result['reason']}

                #### 등급: {grade}
                """


elif  st.session_state["curr_page"] == "speaking__describe_img":
    topic_info = speaking_topic_to_topic_info_map[st.session_state.curr_topic]
    st.title(topic_info['display_name'])

    # random 하게 질문 하나 가져오기
    @st.cache_data
    def load_speaking__describe_img():
        df = pd.read_csv("./data/speaking__describe_img/desc_img.csv")
        return df

    df = load_speaking__describe_img()

    if "img_path" not in st.session_state.exam_context:
        sample = df.sample(n=1).iloc[0]

        img_path = sample["img_path"]
        desc = sample["desc"]

        st.session_state.exam_context["img_path"] = img_path
        st.session_state.exam_context["desc"] = desc
        st.session_state.exam_context["recognized_text"] = ""

    st.image(st.session_state.exam_context['img_path'])
    
    with st.container(border=True):
        recognized_text = recognize_speech()
        if recognized_text:
            st.session_state.exam_context["recognized_text"] = recognized_text
        st.write(st.session_state.exam_context["recognized_text"])

    submit = st.button("제출하기")

    if submit:
        def get_speaking__describe_img(user_input, ref):
            model = ChatOpenAI(
                model="gpt-5.6-luna", temperature=0.8, http_socket_options=()
            ) # CoT 는 다양한 샘플을 만들어야하기 때문에 temperature를 올려야함
            class Evaluation(BaseModel):
                score: int = Field(description="사진 묘사하기 표현 표현 점수. 0~10점")
                feedback: str = Field(description="사진 묘사하기를 더 잘 할 수 있도록하는 자세한 피드백. Markdown형식, 한국어로.")
            parser = JsonOutputParser(pydantic_object=Evaluation)
            format_instructions = parser.get_format_instructions()

            human_prompt_template = HumanMessagePromptTemplate.from_template(
                            "사진 묘사하기 영어 시험이다. 사용자의 응답을 Reference와 비교하여 평가하라.\n사용자: {input}\Reference: {ref}\n{format_instructions}",
                                        partial_variables={"format_instructions": format_instructions})

            prompt = ChatPromptTemplate.from_messages(
                                                    [
                                                        human_prompt_template,
                                                    ])
            eval_chain = prompt | model | parser

            result = eval_chain.invoke({"input": user_input, "ref": ref})
            return result


        st.title("결과 & 피드백- 사진 묘사하기")

        with st.spinner("결과 & 피드백 생성중..."):

            result = get_speaking__describe_img(user_input=st.session_state.exam_context["recognized_text"],
                                                ref=st.session_state.exam_context['desc'])
        
            grade = ""
            if result['score'] >= 8:
                grade = "고급"
            elif 4 < result['score'] < 8:
                grade = "중급"
            elif result['score'] <= 4:
                grade = "초급"

            grade = f"{grade} ({result['score']}/10)"

            f"""
            당신이 제공한 답변은 스피킹 사진 묘사 시험에서 `{grade}` 수준으로 시작하기 좋은 접근입니다.
            
            여기 몇가지 피드백을 드립니다.

            {result['feedback']}
            """


elif  st.session_state["curr_page"] == "speaking__describe_charts":
    topic_info = speaking_topic_to_topic_info_map[st.session_state.curr_topic]
    st.title(topic_info['display_name'])

    # random 하게 질문 하나 가져오기
    @st.cache_data
    def load_speaking__describe_charts():
        df = pd.read_csv("./data/speaking__describe_charts/desc_charts.csv")
        return df

    df = load_speaking__describe_charts()

    if "img_path" not in st.session_state.exam_context:
        sample = df.sample(n=1).iloc[0]

        img_path = sample["img_path"]
        desc = sample["desc"]

        st.session_state.exam_context["img_path"] = img_path
        st.session_state.exam_context["desc"] = desc
        st.session_state.exam_context["recognized_text"] = ""

    st.image(st.session_state.exam_context['img_path'])
    
    with st.container(border=True):
        recognized_text = recognize_speech()
        if recognized_text:
            st.session_state.exam_context["recognized_text"] = recognized_text
        st.write(st.session_state.exam_context["recognized_text"])

    submit = st.button("제출하기")

    if submit:
        def get_speaking__describe_img(user_input, ref):
            model = ChatOpenAI(
                model="gpt-5.6-luna", temperature=0.8, http_socket_options=()
            ) # CoT 는 다양한 샘플을 만들어야하기 때문에 temperature를 올려야함
            class Evaluation(BaseModel):
                score: int = Field(description="도표 보고 발표하기 점수. 0~10점")
                feedback: str = Field(description="도표 보고 발표하기 점수. Markdown형식, 한국어로.")
            parser = JsonOutputParser(pydantic_object=Evaluation)
            format_instructions = parser.get_format_instructions()

            human_prompt_template = HumanMessagePromptTemplate.from_template(
                            "도표보고 발표하기 영어 시험이다. 사용자의 응답을 Reference와 비교하여 평가하라.\n사용자: {input}\Reference: {ref}\n{format_instructions}",
                                        partial_variables={"format_instructions": format_instructions})

            prompt = ChatPromptTemplate.from_messages(
                                                    [
                                                        human_prompt_template,
                                                    ])
            eval_chain = prompt | model | parser

            result = eval_chain.invoke({"input": user_input, "ref": ref})
            return result


        st.title("결과 & 피드백- 도표 보고 발표하기")

        with st.spinner("결과 & 피드백 생성중..."):

            result = get_speaking__describe_img(user_input=st.session_state.exam_context["recognized_text"],
                                                ref=st.session_state.exam_context['desc'])
        
            grade = ""
            if result['score'] >= 8:
                grade = "고급"
            elif 4 < result['score'] < 8:
                grade = "중급"
            elif result['score'] <= 4:
                grade = "초급"

            grade = f"{grade} ({result['score']}/10)"

            f"""
            당신이 제공한 답변은 스피킹 사진 묘사 시험에서 `{grade}` 수준으로 시작하기 좋은 접근입니다.
            
            여기 몇가지 피드백을 드립니다.

            {result['feedback']}
            """


elif  st.session_state["curr_page"] == "writing__dictation":
    from utils import grade_dictation

    topic_info = writing_topic_to_topic_info_map[st.session_state.curr_topic]
    st.title(topic_info['display_name'])

    # random 하게 질문 하나 가져오기
    @st.cache_data
    def load_writing__dictation():
        df = pd.read_csv(APP_DIR / "data/writing__dictation/sent_and_audio.csv")
        return df

    df = load_writing__dictation()

    if "sentence" not in st.session_state.exam_context:
        sample = df.sample(n=1).iloc[0]

        sentence = sample["sentence"]
        audio_file_path = APP_DIR / sample["audio_file_path"]

        st.session_state.exam_context["sample"] = sample
        st.session_state.exam_context["sentence"] = sentence
        st.session_state.exam_context["audio_file_path"] = audio_file_path


    if st.button("시험 시작", type="primary"):
        st.session_state.exam_context["exam_start"] = True
        st.session_state.exam_context["do_speech"] = True

    if st.session_state.exam_context.get("exam_start", False):
        if st.session_state.exam_context["do_speech"]:
            autoplay_audio(st.session_state.exam_context["audio_file_path"])
            st.session_state.exam_context["do_speech"] = False

        if st.button("다시 듣기"):
            autoplay_audio(st.session_state.exam_context["audio_file_path"])

        with st.form("dictation_form"):
            user_answer = st.text_input(
                "들은 문장을 영어로 입력하세요.",
                placeholder="재생된 문장을 받아쓰세요.",
            )
            submitted = st.form_submit_button("제출하고 채점하기", type="primary")

        if submitted:
            user_answer = user_answer.strip()
            if not user_answer:
                st.warning("답안을 입력한 뒤 제출해 주세요.")
            else:
                st.session_state.exam_context["user_answer"] = user_answer
                with st.spinner("채점 중..."):
                    model_result = evaluate_writing_answer(
                        instruction=(
                            "사용자의 받아쓰기 답안을 정답 문장과 비교하여 정확성을 평가하라. "
                            "누락, 추가, 철자, 문법과 문장부호를 고려하라."
                        ),
                        context=st.session_state.exam_context["sentence"],
                        user_answer=user_answer,
                    )
                    automatic_result = grade_dictation(
                        correct_script=st.session_state.exam_context["sentence"],
                        student_response=user_answer,
                    )

                automatic_score = automatic_result["accuracy"] * 10
                st.session_state.exam_context["result"] = {
                    "reason": model_result["reason"],
                    "feedback": model_result["feedback"],
                    "model_score": model_result["score"],
                    "automatic_score": automatic_score,
                    "final_score": (model_result["score"] + automatic_score) / 2,
                }

        if "result" in st.session_state.exam_context:
            result = st.session_state.exam_context["result"]
            with st.container(border=True):
                st.subheader("평가 결과")
                st.markdown(
                    f"**정답:** {st.session_state.exam_context['sentence']}  \n"
                    f"**내 답안:** {st.session_state.exam_context['user_answer']}"
                )
                st.metric("총점", f"{result['final_score']:.1f} / 10")
                st.markdown(result["reason"])
                st.markdown(f"**개선 피드백**  \n{result['feedback']}")
                st.caption(
                    f"모델 평가 {result['model_score']}/10 · "
                    f"문자열 정확도 {result['automatic_score']:.1f}/10"
                )


elif st.session_state["curr_page"] in WRITING_EXAM_CONFIGS:
    config = WRITING_EXAM_CONFIGS[st.session_state["curr_page"]]
    topic_info = writing_topic_to_topic_info_map[st.session_state.curr_topic]
    st.title(topic_info["display_name"])

    if "writing_context" not in st.session_state.exam_context:
        with st.spinner("시험 문제를 생성 중..."):
            st.session_state.exam_context["writing_context"] = generate_writing_context(
                config["generation_prompt"]
            )

    with st.container(border=True):
        st.subheader(config["context_heading"])
        st.markdown(st.session_state.exam_context["writing_context"])

    with st.form(f"{st.session_state['curr_page']}_form"):
        user_answer = st.text_area(
            config["answer_label"],
            placeholder=config["answer_placeholder"],
            height=240,
        )
        submitted = st.form_submit_button("제출하고 채점하기", type="primary")

    if submitted:
        user_answer = user_answer.strip()
        if not user_answer:
            st.warning("답안을 입력한 뒤 제출해 주세요.")
        else:
            st.session_state.exam_context["user_answer"] = user_answer
            with st.spinner("답안을 평가 중..."):
                st.session_state.exam_context["result"] = evaluate_writing_answer(
                    instruction=config["evaluation_instruction"],
                    context=st.session_state.exam_context["writing_context"],
                    user_answer=user_answer,
                )

    if "result" in st.session_state.exam_context:
        result = st.session_state.exam_context["result"]
        with st.container(border=True):
            st.subheader("평가 결과")
            st.metric("점수", f"{result['score']} / 10")
            st.markdown(result["reason"])
            st.markdown(f"**개선 피드백**  \n{result['feedback']}")
