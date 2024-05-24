import requests
import plugins
from plugins import *
from bridge.context import ContextType
from bridge.reply import Reply, ReplyType
from common.log import logger
from gradio_client import Client
import os

GRADIO_API = "openbmb/MiniCPM-Llama3-V-2_5"
UPLOAD_DIR = 'uploaded_images'

@plugins.register(name="ImageAnalysis",
                  desc="上传图片并进行分析",
                  version="1.0",
                  author="Cool",
                  desire_priority=100)
class ImageAnalysis(Plugin):
    def __init__(self):
        super().__init__()
        self.handlers[Event.ON_HANDLE_CONTEXT] = self.on_handle_context
        logger.info(f"[{__class__.__name__}] inited")
        self.client = Client(GRADIO_API)
        if not os.path.exists(UPLOAD_DIR):
            os.makedirs(UPLOAD_DIR)

    def get_help_text(self, **kwargs):
        help_text = f"发送图片进行分析"
        return help_text

    def on_handle_context(self, e_context: EventContext):
        if e_context['context'].type != ContextType.IMAGE:
            return

        image_file = e_context['context'].content
        image_path = os.path.join(UPLOAD_DIR, image_file.name)

        # 保存用户上传的图片
        with open(image_path, 'wb') as f:
            f.write(image_file.read())
        
        logger.info(f"[{__class__.__name__}] 收到图片消息")
        reply = Reply()
        reply.type = ReplyType.TEXT
        reply.content = "图片上传成功"
        e_context["reply"] = reply
        e_context.action = EventAction.BREAK_PASS

        # 分析图片并返回结果
        analysis_result = self.analyze_image(image_path)
        if analysis_result:
            reply.type = ReplyType.TEXT
            reply.content = analysis_result
            e_context["reply"] = reply
            e_context.action = EventAction.BREAK_PASS
        else:
            reply.type = ReplyType.ERROR
            reply.content = "图片分析失败，请稍后再试"
            e_context["reply"] = reply
            e_context.action = EventAction.BREAK_PASS

    def analyze_image(self, image_path):
        try:
            upload_result = self.client.predict(
                image=image_path,
                _chatbot=[],
                api_name="/upload_img"
            )
            logger.info(f"Image upload result: {upload_result}")
            
            question_result = self.client.predict(
                _question="请详细且充分的描述图像内容及相关信息",
                _chat_bot=[],
                params_form="Sampling",
                num_beams=3,
                repetition_penalty=1.2,
                repetition_penalty_2=1.05,
                top_p=0.8,
                top_k=100,
                temperature=0.7,
                api_name="/respond"
            )
            logger.info(f"Question result: {question_result}")
            return question_result
        except Exception as e:
            logger.error(f"分析图片时发生错误：{e}")
            return None

if __name__ == "__main__":
    image_analysis_plugin = ImageAnalysis()
