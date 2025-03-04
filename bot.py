import asyncio
import os
import sys
import argparse
import json
import  base64

from pipecat.services.azure import AzureLLMService, AzureSTTService, AzureTTSService, Language





from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.frames.frames import LLMMessagesFrame, EndFrame
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.openai import OpenAILLMService
from pipecat.services.elevenlabs import ElevenLabsTTSService
from pipecat.services.cartesia import CartesiaTTSService
from pipecat.transports.services.daily import DailyParams, DailyTransport
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.frames.frames import (
    OutputImageRawFrame,
    SpriteFrame,
    Frame,
    LLMMessagesFrame,
    TTSAudioRawFrame,
    TTSStoppedFrame,
)

from PIL import Image

from prompt import prompt

from loguru import logger

from dotenv import load_dotenv

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

daily_api_key = os.getenv("DAILY_API_KEY", "")
daily_api_url = os.getenv("DAILY_API_URL", "https://api.daily.co/v1")

# sprites = []

# script_dir = os.path.dirname(__file__)

# for i in range(1, 21):
#     # Build the full path to the image file
#     full_path = os.path.join(script_dir, f"assets/{i}.png")
#     # Get the filename without the extension to use as the dictionary key
#     # Open the image and convert it to bytes
#     with Image.open(full_path) as img:
#         sprites.append(OutputImageRawFrame(image=img.tobytes(), size=img.size, format=img.format))

# flipped = sprites[::-1]
# sprites.extend(flipped)

# # When the bot isn't talking, show a static image of the cat listening
# quiet_frame = sprites[0]
# talking_frame = SpriteFrame(images=sprites)

# class TalkingAnimation(FrameProcessor):
#     """
#     This class starts a talking animation when it receives an first AudioFrame,
#     and then returns to a "quiet" sprite when it sees a TTSStoppedFrame.
#     """

#     def __init__(self):
#         super().__init__()
#         self._is_talking = False

#     async def process_frame(self, frame: Frame, direction: FrameDirection):
#         await super().process_frame(frame, direction)

#         if isinstance(frame, TTSAudioRawFrame):
#             if not self._is_talking:
#                 await self.push_frame(talking_frame)
#                 self._is_talking = True
#         elif isinstance(frame, TTSStoppedFrame):
#             await self.push_frame(quiet_frame)
#             self._is_talking = False

#         await self.push_frame(frame)


#**********




# script_dir = os.path.dirname(__file__)
# # image_path = os.path.join("") # Using just the first image
# with Image.open("/home/atish41/newtest/img.png") as img:
#     bot_frame = OutputImageRawFrame(image=img.tobytes(), size=img.size, format=img.format)

# class StaticImageProcessor(FrameProcessor):
#     """
#     This class maintains a static image throughout the conversation,
#     regardless of whether the bot is talking or not.
#     """
#     async def process_frame(self, frame: Frame, direction: FrameDirection):
#         await super().process_frame(frame, direction)
#         # Always push the frame through
#         await self.push_frame(frame)

# class StaticImageProcessor(FrameProcessor):
#     async def process_frame(self, frame: Frame, direction: FrameDirection):
#         await super().process_frame(frame, direction)
#         await self.push_frame(frame)


async def main(room_url: str, token: str):
    transport = DailyTransport(
        room_url,
        token,
        "Paddi AI",
        DailyParams(
            api_url=daily_api_url,
            api_key=daily_api_key,
            audio_in_enabled=True,
            audio_out_enabled=True,
            camera_out_enabled=False,
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
            transcription_enabled=True,
        ),
    )

    # config_str = base64.b64decode(config_b64).decode()
    # config = json.loads(config_str)


    tts_params = CartesiaTTSService.InputParams(
            speed="normal",
            emotion=["positivity:high", "curiosity"]
        )
    
    # tts = ElevenLabsTTSService(api_key=os.getenv("ELEVENLABS_API_KEY"), voice_id=os.getenv("ELEVENLABS_VOICE_ID"))
    	
    # SYNTH_CONFIG=AzureSynthesizerConfig.from_telephone_output_device( voice_name="en-NG-EzinneNeural", language_code="en-NG", )
    # Configure service
    # Configure service
    tts_service = AzureTTSService(
        api_key=os.getenv("AZURE_API_KEY"),
        region=os.getenv("AZURE_REGION"),
        voice="en-NG-AbeoNeural",
        # voice="en-NG-EzinneNeural",
        params=AzureTTSService.InputParams(
            language=Language.EN_US,
            rate="1.1",
            style="cheerful"
        )
    )



    # tts = CartesiaTTSService(
    #         api_key=os.getenv("CARTESIA_API_KEY"),
    #         # voice_id='a0e99841-438c-4a64-b679-ae501e7d6091',
    #         voice_id='bxPMdBTxMI0LMo67TDEK',
    #         params=tts_params
    #     )

    llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o")

    messages = [
        {
            "role": "system",
            "content":prompt,
        },
    ]

    context = OpenAILLMContext(messages)
    context_aggregator = llm.create_context_aggregator(context)
    # ts=TalkingAnimation()

    pipeline = Pipeline(
        [
            transport.input(),
            context_aggregator.user(),
            llm,
            tts_service,
            transport.output(),
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(pipeline, PipelineParams(allow_interruptions=True))


    # @transport.event_handler("on_first_participant_joined")
    # async def on_first_participant_joined(transport, participant):
    #     await transport.capture_participant_transcription(participant["id"])
    #     await task.queue_frames([LLMMessagesFrame(messages)])

    @transport.event_handler("on_participant_joined")
    async def on_participant_joined(transport, participant):
        await transport.capture_participant_transcription(participant["id"])
        await task.queue_frames([LLMMessagesFrame(messages)])

    @transport.event_handler("on_participant_left")
    async def on_participant_left(transport, participant, reason):
        await task.queue_frame(EndFrame())

    @transport.event_handler("on_call_state_updated")
    async def on_call_state_updated(transport, state):
        if state == "left":
            await task.queue_frame(EndFrame())

    runner = PipelineRunner()

    await runner.run(task)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipecat Bot")
    parser.add_argument("-u", required=True,type=str, help="Room URL")
    parser.add_argument("-t",  required=True,type=str, help="Token")
    # parser.add_argument("--config", required=True, help="Base64 encoded configuration")
    args = parser.parse_args()

    asyncio.run(main(args.u, args.t ))