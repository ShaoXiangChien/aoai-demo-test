import streamlit as st
from streamlit_elements import mui, elements
import uuid


def message(text, key="", seed="", avatar_style="", is_user=False, is_openai=False, add_openai=False):
    if is_user:
        with elements(str(uuid.uuid4())):
            mui.Box(
                mui.Avatar(
                    src="https://www.pngitem.com/pimgs/m/22-220721_circled-user-male-type-user-colorful-icon-png.png", alt="Azure bot", ),
                sx={
                    "margin": "2% 0",
                    "display": "flex",
                    "flexDirection": "row-reverse"
                },
            )
            mui.Box(
                mui.Box(text, sx={
                    "marginRight": "5%",
                    "borderRadius": "15px",
                    "border": "solid rgba(39,39,49,255)",
                    "width": "fit-content",
                    "padding": "10px",
                    "backgroundColor": "rgba(39,39,49,255)",
                }, ),
                sx={
                    "margin": "2% 0",
                    "display": "flex",
                    "flexDirection": "row-reverse"
                },
            )
    elif add_openai:
        with elements(str(uuid.uuid4())):
            mui.Stack(
                mui.Avatar(
                    src="https://cdn3.iconfinder.com/data/icons/chat-bot-emoji-filled-color/300/141453384Untitled-3-512.png", alt="Azure bot", sx={"border": "solid 2px #AEA3CD"}, ),
                mui.Avatar(
                    src="https://a.fsdn.com/allura/s/openai-codex/icon?79624fcb195fd94801a8f821064006313d5b3e3dc73ecf53843d99ec566053c6?&w=120", alt="openai", ),
                sx={
                    "margin": "2% 0",
                    "alignItems": "center"
                },
                direction="row",
                spacing=1,

            )
            mui.Box(text, sx={
                "marginLeft": "5%",
                "borderRadius": "15px",
                "border": "solid rgba(39,39,49,255)",
                "width": "fit-content",
                "padding": "10px",
                "backgroundColor": "rgba(39,39,49,255)",
            }, )
    elif is_openai:
        with elements(str(uuid.uuid4())):
            mui.Stack(
                mui.Avatar(
                    src="https://a.fsdn.com/allura/s/openai-codex/icon?79624fcb195fd94801a8f821064006313d5b3e3dc73ecf53843d99ec566053c6?&w=120", alt="openai", ),
                sx={
                    "margin": "2% 0",
                    "alignItems": "center"
                },
                direction="row",
                spacing=1,

            )
            mui.Box(text, sx={
                "marginLeft": "5%",
                "borderRadius": "15px",
                "border": "solid rgba(39,39,49,255)",
                "width": "fit-content",
                "padding": "10px",
                "backgroundColor": "rgba(39,39,49,255)",
            }, )
    else:
        with elements(str(uuid.uuid4())):
            mui.Stack(
                mui.Avatar(
                    src="https://cdn3.iconfinder.com/data/icons/chat-bot-emoji-filled-color/300/141453384Untitled-3-512.png", alt="Azure bot", sx={"border": "solid 2px #AEA3CD"}, ),
                sx={
                    "margin": "2% 0",
                    "alignItems": "center"
                },
                direction="row",
                spacing=1,

            )
            mui.Box(text, sx={
                "marginLeft": "5%",
                "borderRadius": "15px",
                "border": "solid rgba(39,39,49,255)",
                "width": "fit-content",
                "padding": "10px",
                "backgroundColor": "rgba(39,39,49,255)",
            }, )

# def user_message(text):
#     with elements("new_elements"):
#         mui.Box(
#             mui.Avatar(
#                 src="https://www.pngitem.com/pimgs/m/22-220721_circled-user-male-type-user-colorful-icon-png.png", alt="Azure bot", ),
#             sx={
#                 "margin": "2% 0",
#                 "display": "flex",
#                 "flexDirection": "row-reverse"
#             },
#         )
#         mui.Box(
#             mui.Typography(text, sx={
#                 "marginRight": "5%",
#                 "borderRadius": "15px",
#                 "border": "solid rgba(39,39,49,255)",
#                 "width": "fit-content",
#                 "padding": "10px",
#                 "backgroundColor": "rgba(39,39,49,255)",
#             }, key=str(uuid.uuid4())),
#             sx={
#                 "margin": "2% 0",
#                 "display": "flex",
#                 "flexDirection": "row-reverse"
#             },
#         )


def msg_btns(btns):
    with elements(str(uuid.uuid4())):
        mui.Stack(
            mui.Button(btns[0], variant="contained",
                       sx={'backgroundColor': "gray"}),
            mui.Button(btns[1], variant="contained",
                       sx={'backgroundColor': "gray"}),
            mui.Button(btns[2], variant="contained",
                       sx={'backgroundColor': "gray"}),
            width="fit-content",
            spacing=1,
            sx={
                "marginLeft": "5%",
            }
        )


if __name__ == "__main__":
    message("hello, I'm chat bot")
    msg_btns(["1412424214", '2342141234141', '2134214243'])
    # message("你是誰", is_user=True)
