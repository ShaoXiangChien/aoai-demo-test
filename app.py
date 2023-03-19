import streamlit as st
import streamlit_authenticator as stauth
from component import message
import requests
import openai
from openai.embeddings_utils import get_embedding, cosine_similarity
import pandas as pd
import numpy as np
import azure.cognitiveservices.speech as speechsdk


openai.api_type = 'azure'
openai.api_version = "2022-12-01"
openai.api_key = st.secrets['api']['openai_key']
openai.api_base = st.secrets['api']['openai_base']
system_message_template = "<|im_start|>system\n{}\n<|im_end|>"
system_message = system_message_template.format("你是專門負責摘要的機器人")

st.set_page_config(
    page_title="OpenAI Chat - Demo",
)
authenticator = stauth.Authenticate(
    {
        "usernames": {"mtcuser": st.secrets['mtcuser']}
    },
    st.secrets['cookie']['name'],
    st.secrets['cookie']['key'],
    st.secrets['cookie']['expiry_days'],
)


st.header("OpenAI Chat - Demo")
model = st.sidebar.selectbox("Model", ['davinci003', 'gpt-35-turbo'])
choice = st.sidebar.selectbox(
    "Select chat mode", ["Original QnA", "AOAI embeddings"])
st.subheader(choice)
st.markdown(
    """
    <style>

        div[data-testid="column"]:nth-of-type(2)
        {
            text-align: end;
        } 
    </style>
    """, unsafe_allow_html=True
)

container = st.container()


def start_conversation():
    endpoint = "https://directline.botframework.com/v3/directline/conversations"
    response = requests.post(endpoint, headers=st.session_state['headers'])
    if response.status_code == 201:
        res = response.json()
        st.session_state['conversationId'] = res['conversationId']

        resp = GET()


def POST(payload):
    API_URL = f"https://directline.botframework.com/v3/directline/conversations/{st.session_state['conversationId']}/activities"

    response = requests.post(
        API_URL, headers=st.session_state['headers'], json=payload)
    return response.json()


def GET():
    API_URL = f"https://directline.botframework.com/v3/directline/conversations/{st.session_state['conversationId']}/activities"
    response = requests.get(API_URL, headers=st.session_state['headers'])
    return response.json()


def org_query(payload, aoai_enrich):
    with container:
        if 'activities' not in st.session_state:
            message(payload['text'], is_user=True,
                    avatar_style="adventurer-neutral", key="usr_0")
        else:
            for idx, turn in enumerate(st.session_state['activities']):
                if 'text' not in turn:
                    message("抱歉，我無法回答這個問題", key=str(idx + 1), seed="Felix")
                    continue
                if turn['from']['id'] != "user1":
                    if 'tone' in turn:
                        col1, col2 = st.columns(2)

                        col2.write(f"Prompt: 使用{turn['tone']}口吻改寫")

                    message(turn['text'], key=str(idx + 1),
                            seed="Felix", add_openai=True if turn.get('aoai') else False)
                else:
                    message(turn['text'], is_user=True, avatar_style="adventurer-neutral",
                            key="usr_" + str(idx + 1))
            message(payload['text'], is_user=True, avatar_style="adventurer-neutral", key="usr_" +
                    str(len(st.session_state['activities'])+1))
            pass

    POST(payload)
    resp = GET()
    act_len = len(st.session_state['activities'])
    new_activities = resp['activities'][act_len+1:]
    if aoai_enrich:
        tone = st.sidebar.selectbox(
            "選擇一種口吻", ['專家', '小孩', '活潑', '工程師', '業務'], key="")
        with container:
            col1, col2 = st.columns(2)

            col2.write(f"Prompt: 使用{tone}口吻改寫")

        for idx, act in enumerate(new_activities):
            if act['from']['id'] != 'user1' and 'text' in act:
                new_activities[idx]['text'] = aoai_enrichment(
                    act['text'], tone)
                new_activities[idx]['tone'] = tone
                new_activities[idx]['aoai'] = True
    st.session_state['activities'] += new_activities

    return resp


def create_prompt(messages):
    prompt = system_message
    message_template = "\n<|im_start|>{}\n{}\n<|im_end|>"
    for message in messages:
        prompt += message_template.format(message['sender'], message['text'])
    prompt += "\n<|im_start|>assistant\n"
    return prompt


def summarize(prompt):
    if model == "gpt-35-turbo":
        message = [{"sender": "user", "text": f"""
        '''
        {prompt}
        '''

        請用專業的口吻來簡化以上文字，且不要用本文開頭

        '''
        """}]
        summary = openai.Completion.create(
            engine="gpt-35-turbo",
            prompt=create_prompt(message),
            temperature=0.7,
            max_tokens=800,
            top_p=0.91,
            stop=["<|im_end|>"])["choices"][0]["text"]
        # st.success("done")
    else:

        augmented_prompt = f"請用以專業的口吻摘要這段文字: {prompt}"

        summary = openai.Completion.create(
            engine="davinci003",
            prompt=augmented_prompt,
            temperature=.5,
            max_tokens=2000,
        )["choices"][0]["text"].replace("\n", "")

    return summary


def aoai_enrichment(ans, tone):
    augmented_prompt = f"請用中文以{tone}的口吻改寫這段文字: {ans}"
    summary = openai.Completion.create(
        engine="davinci003",
        prompt=augmented_prompt,
        temperature=.5,
        max_tokens=2000,
    )["choices"][0]["text"].replace("\n", "")

    return summary


def aoai_query(query, mode, aoai_enrichment):
    if aoai_enrichment:
        docs = search_docs(query, mode, top_n=3)
        return summarize("".join(docs))
    else:
        docs = search_docs(query, mode, top_n=1)
        return docs[0]


def get_text():
    input_text = st.text_input("", "", key="text", label_visibility='hidden')
    return input_text


def clear_text():
    st.session_state['text'] = ""

# search through the reviews for a specific product


def search_docs(user_query, mode, top_n=3):
    if user_query != "":
        embedding = get_embedding(
            user_query,
            engine="text-similarity-curie-001"
        )
        st.write("Embedding Length: 4096")
        st.write("Sample of embeddings")
        st.write(embedding[:5])

        st.session_state['qa_df']["similarities"] = st.session_state['qa_df'][f'{mode[:-1]}_embedding'].apply(
            lambda x: cosine_similarity(x, embedding))

        res = (
            st.session_state['qa_df'].sort_values(
                "similarities", ascending=False)
            .head(top_n)
        )
        if top_n == 1:
            st.session_state['top_score'] = res.iloc[0]['similarities']
            st.session_state['top_q'] = res.iloc[0]['Question']
        return res['Answer'].to_list()


def load_data(mode):
    st.session_state['qa_df'] = pd.read_csv("./COVID-FAQ_qnas.csv")
    tmp = np.load(f"./{mode[:-1]}_embeddings.npy")
    st.session_state['qa_df'][f'{mode[:-1]}_embedding'] = tmp.tolist()


def read():
    clear_text()
    speech_synthesis_result = st.session_state['speech_synthesizer'].speak_text_async(
        st.session_state['read_text']).get()
    if speech_synthesis_result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        print("Speech synthesized for text [{}]".format(
            st.session_state['read_text']))
    elif speech_synthesis_result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = speech_synthesis_result.cancellation_details
        print("Speech synthesis canceled: {}".format(
            cancellation_details.reason))
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            if cancellation_details.error_details:
                print("Error details: {}".format(
                    cancellation_details.error_details))
                print("Did you set the speech resource key and region values?")


def main():
    globals()
    name, authentication_status, username = authenticator.login(
        'Login', 'main')
    if authentication_status:
        authenticator.logout('Logout', 'main')
        st.write('Welcome *%s*' % (name))
#     if 'cs_key' in st.session_state:
#         speech_config = speechsdk.SpeechConfig(
#             subscription=st.session_state['cs_key'], region="eastus")
#         audio_config = speechsdk.audio.AudioOutputConfig(
#             use_default_speaker=True)
#         speech_config.speech_synthesis_voice_name = 'zh-CN-YunxiNeural'

#         st.session_state['speech_synthesizer'] = speechsdk.SpeechSynthesizer(
#             speech_config=speech_config)

        st.session_state['headers'] = {
            "Authorization": f"Bearer {st.secrets['api']['qna_bot_key']}"}

        if choice == "Original QnA":
            aoai_enrich = st.sidebar.checkbox("AOAI Tone-Enhanced")

            if 'activities' not in st.session_state:
                st.session_state['activities'] = []

            with container:
                message(
                    "我是Covid FAQ機器人，有什麼我可以協助您的嗎?", key="0", seed="Felix")

            user_input = get_text()

            if user_input != "":
                if 'conversationId' not in st.session_state:
                    start_conversation()

                org_query({
                    "type": "message",
                    "from": {
                        "id": "user1"
                    },
                    "text": user_input
                }, aoai_enrich)

                if 'activities' in st.session_state:
                    with container:

                        # for idx, turn in enumerate(st.session_state['activities'][2:]):
                        #     if turn['from']['id'] != "user1":
                        #         message(turn['text'])
                        #     else:
                        #         message(turn['text'], is_user=True)
                        try:
                            message(st.session_state['activities'][-1]['text'],
                                    key=str(len(st.session_state['activities'])+1), seed="Felix", add_openai=True if st.session_state['activities'][-1].get("aoai") else False)
                            st.session_state['read_text'] = st.session_state['activities'][-1]['text']

                        except Exception as e:
                            # st.write(st.session_state['activities'][-1])

                            if 'attachments' in st.session_state['activities'][-1]:
                                attachment = st.session_state['activities'][-1]['attachments'][0]
                                with container:
                                    message(attachment['content']['title'], key=str(
                                        len(st.session_state['activities'])+1), seed="Felix")
                                    btns = attachment['content']['buttons']
                                    for btn in btns:
                                        st.write(btn['value'])
                                        # with elements(str(uuid.uuid4())):

                                        # org_query({
                                        #     "type": "message",
                                        #     "from": {
                                        #           "id": "user1"
                                        #           },
                                        #     "text": btn['value']
                                        # }, aoai_enrich)

                            else:
                                st.write(st.session_state['activities'][-1])
                                message("抱歉，我無法回答這個問題", key=str(
                                    len(st.session_state['activities'])+1), seed="Felix")
                                st.session_state['read_text'] = "抱歉，我無法回答這個問題"

    #                     st.button("speech", on_click=read)

            elif 'activities' in st.session_state:
                with container:
                    for idx, turn in enumerate(st.session_state['activities']):
                        if 'text' not in turn:
                            message("抱歉，我無法回答這個問題", key=str(
                                idx + 1), seed="Felix")
                            if idx == len(st.session_state['activities']) - 1:
                                st.session_state['read_text'] = "抱歉，我無法回答這個問題"
    #                             st.button("speech", on_click=read)
                            continue

                        if turn['from']['id'] != "user1":
                            message(turn['text'], key=str(
                                idx + 1), seed="Felix", add_openai=True if turn.get('aoai') else False)
                            if idx == len(st.session_state['activities']) - 1:
                                st.session_state['read_text'] = st.session_state['activities'][-1]['text']
    #                             st.button("speech", on_click=read)
                        else:
                            message(turn['text'], is_user=True, avatar_style="adventurer-neutral",
                                    key="usr_" + str(idx + 1))
        else:
            # qa_mode = st.sidebar.radio("Select search target", [
            #                            'answers', 'questions'])
            qa_mode = "questions"
            st.sidebar.write("""
            藉由Text Embeddings來強化搜尋比對的能力
            """)
            aoai_enrich = st.sidebar.checkbox("AOAI Enrichment")
            user_input = ""
            if "qa_df" not in st.session_state:
                load_data(qa_mode)
                st.session_state['qa_mode'] = qa_mode

            if qa_mode != st.session_state.get('qa_mode'):
                load_data(qa_mode)
                st.session_state['qa_mode'] = qa_mode

            if 'generated' not in st.session_state:
                st.session_state['generated'] = []

            if 'past' not in st.session_state:
                st.session_state['past'] = []

            with container:
                message(
                    "我是整合了AOAI的Covid FAQ機器人，有什麼我可以協助您的嗎?", key="0", seed="Aneka", is_openai=True)

                if st.session_state['generated']:

                    for i in range(len(st.session_state['generated'])):
                        message(st.session_state['past'][i],
                                is_user=True, avatar_style="adventurer-neutral", key=str(i+1) + '_user')
                        message(st.session_state["generated"]
                                [i], key=str(i+1),  seed="Aneka", is_openai=True)

            user_input = get_text()
            if user_input != "":
                with container:
                    st.session_state.past.append(user_input)
                    message(user_input, is_user=True, avatar_style="adventurer-neutral", key=str(
                        len(st.session_state.past))+"_user")

                res = aoai_query(user_input, qa_mode, aoai_enrich)

                with container:
                    st.session_state.generated.append(res)
                    message(res, key=str(
                        len(st.session_state.generated)), seed="Aneka", is_openai=True)
                    if not aoai_enrich:
                        st.write(
                            f"Selected Question: {st.session_state['top_q']}")
                        st.write(
                            f"Similarity Score: {st.session_state['top_score']:.2f}")

    elif authentication_status == False:
        st.error('Username/password is incorrect')
    elif authentication_status == None:
        st.warning('Please enter your username and password')


if __name__ == "__main__":
    main()


# if st.session_state['generated']:

#     for i in range(len(st.session_state['generated'])-1, -1, -1):
#         message(st.session_state["generated"][i], key=str(i))
#         message(st.session_state['past'][i],
#                 is_user=True, key=str(i) + '_user')
