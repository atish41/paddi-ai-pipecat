import asyncio
import os
import sys
import argparse
from PIL import Image
from loguru import logger
from dotenv import load_dotenv

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.frames.frames import (
    OutputImageRawFrame,
    SpriteFrame,
    Frame,
    LLMMessagesFrame,
    TTSAudioRawFrame,
    TTSStoppedFrame,
    EndFrame
)
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.openai import OpenAILLMService
from pipecat.services.elevenlabs import ElevenLabsTTSService
from pipecat.services.cartesia import CartesiaTTSService
from pipecat.transports.services.daily import DailyParams, DailyTransport
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from prompt import prompt

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

daily_api_key = os.getenv("DAILY_API_KEY", "")
daily_api_url = os.getenv("DAILY_API_URL", "https://api.daily.co/v1")

# Initialize sprite loading
sprites = []
script_dir = os.path.dirname(__file__)

# Load sprites
for i in range(1, 26):  # Adjust range based on your number of images
    full_path = os.path.join(script_dir, f"assets/{i}.png")
    try:
        with Image.open(full_path) as img:
            sprites.append(OutputImageRawFrame(image=img.tobytes(), size=img.size, format=img.format))
            logger.debug(f"Successfully loaded sprite: {full_path}")
    except Exception as e:
        logger.error(f"Failed to load sprite {full_path}: {str(e)}")

# Create animation sequence
flipped = sprites[::-1]
sprites.extend(flipped)

# Define static frames
quiet_frame = sprites[0] if sprites else None
talking_frame = SpriteFrame(images=sprites)

class TalkingAnimation(FrameProcessor):
    """
    Handles the talking animation state transitions
    """
    def __init__(self):
        super().__init__()
        self._is_talking = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TTSAudioRawFrame):
            if not self._is_talking:
                await self.push_frame(talking_frame)
                self._is_talking = True
        elif isinstance(frame, TTSStoppedFrame):
            await self.push_frame(quiet_frame)
            self._is_talking = False

        await self.push_frame(frame)

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
            camera_out_enabled=True,  # Changed to True for image output
            camera_out_width=1024,    # Added specific dimensions
            camera_out_height=576,    # Added specific dimensions
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
            transcription_enabled=True,
        ),
    )

    tts_params = CartesiaTTSService.InputParams(
        speed="normal",
        emotion=["positivity:high", "curiosity"]
    )
    
    tts = ElevenLabsTTSService(
        api_key=os.getenv("ELEVENLABS_API_KEY"), 
        voice_id=os.getenv("ELEVENLABS_VOICE_ID")
    )

    llm = OpenAILLMService(
        api_key=os.getenv("OPENAI_API_KEY"), 
        model="gpt-4o"
    )

    messages = [
        {
            "role": "system",
            "content": prompt,
        },
    ]

    context = OpenAILLMContext(messages)
    context_aggregator = llm.create_context_aggregator(context)
    ta = TalkingAnimation()

    pipeline = Pipeline(
        [
            transport.input(),
            context_aggregator.user(),
            llm,
            tts,
            ta,
            transport.output(),
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(pipeline, PipelineParams(allow_interruptions=True))
    
    # Initialize with quiet frame
    if quiet_frame:
        await task.queue_frame(quiet_frame)

    @transport.event_handler("on_first_participant_joined")
    async def on_first_participant_joined(transport, participant):
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
    parser.add_argument("-u", required=True, type=str, help="Room URL")
    parser.add_argument("-t", required=True, type=str, help="Token")
    args = parser.parse_args()

    asyncio.run(main(args.u, args.t))