import os
import copy
import asyncio
import aiohttp
import base64
import time
import json
import math
import random

import numpy as np
from typing import List, Optional
from dataclasses import dataclass
from statistics import mean, median
from tqdm import tqdm
from enum import Enum

random.seed(42)

mute = False

class SOS_TOKEN(Enum):
    YES = "是"
    NO = "否"
    NOISE = "1"
    BC = "2"
    SPEECH = "3"

PAYLOAD = {
    "model": None,
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": None,
                },
                {
                    "type": "audio_url",
                    "audio_url": {"url": None},
                }
            ]
        }
    ],
    "stream": False,
    "temperature": 0.0,
    "top_k": 1,
    "repetition_penalty": 1.0,
    "max_completion_tokens": 1,
    "logprobs": True,
    "top_logprobs": 5,
    "stop_token_ids": [151667],
}

HEADERS = {
    "Content-Type": "application/json",
}

prompts = {
    "binary": "请识别电话沟通场景中如下声音片段的话轮转换意图，判断该片段是否包含明确的开始说话信号。请区分以下两种情况：若检测到清晰语音起始或强烈发言意愿（如语句开头、语气转折），应回复<是>；若仅含附和词（如\"嗯\"、\"yeah\"）、非语言声音（如喷嚏、咳嗽、笑声）、噪声或近似静默等非打断性信号，应回复<否>",
    "multi": "请识别电话沟通场景中如下声音片段的类别。若检测到清晰语音起始或强烈发言意愿（如语句开头、语气转折），应回复<3>；否则若含附和词（如\"嗯\"、\"yeah\"），应回复<2>；除1、2外的所有其他情况如非语言声音（如喷嚏、咳嗽、笑声）、噪声或近似静默等非打断性信号，应回复<1>",
}

def dprint(content):
    if mute:
        return
    print(content)

async def kick_model(model_path, input_audios, concurrence):
    PAYLOAD["model"] = model_path
    model_seq = int(os.path.split(model_path)[-1].split('_')[-2])
    model_type = "binary" if model_seq <= 8 else "multi"
    assert model_type in ["binary", "multi"]
    PAYLOAD["messages"][0]["content"][0]["text"] = prompts[model_type]
    sos_client = SOSClient()

    model_outputs = await sos_client(input_audios, concurrence)
    return model_outputs


def response_to_probs(response):
    data = json.loads(response)
    logprobs = data["choices"][0]["logprobs"]['content'][0]["top_logprobs"]
    probs = [math.exp(logprob) for logprob in [item['logprob'] for item in logprobs]]
    probs_norm = [prob / sum(probs) for prob in probs]
    tokens = [item['token'] for item in logprobs]
    probs = dict(zip(tokens, probs_norm))
    return probs


def encode_pcm(pcm):
    audio_b64 = base64.b64encode(pcm.tobytes()).decode("utf-8")
    return f"data:audio/pcm;base64,{audio_b64}"


def distribute_audios(input_audios, division):
    input_audios_sorted = sorted(input_audios, key=lambda x: len(x[1]))
    audios = []
    for idx in range(division):
        input_audios_selected = input_audios_sorted[idx::division]
        random.shuffle(input_audios_selected)
        audios.append([(item[0], encode_pcm(item[1])) for item in input_audios_selected])
    assert sum([len(item) for item in audios]) == len(input_audios)
    return audios


def print_results(results, concurrence, start_time, end_time):
    num_requests = len(results)
    assert num_requests == sum(1 for r in results if r.success)
    latencies = [r.latency for r in results if r.success]

    total_duration = end_time - start_time
    requests_per_second = num_requests / total_duration

    dprint("\n" + "=" * 60)
    dprint("📊 基准测试结果")
    dprint("=" * 60)
    dprint(f"并发数:             {concurrence}")
    dprint(f"总请求数:           {num_requests}")
    dprint(f"测试总时长:         {total_duration:.2f}s")
    dprint(f"请求速率:           {requests_per_second:.2f} req/s")

    if latencies:
        dprint(f"\n⏱️  延迟统计:")
        dprint(f"平均延迟:           {mean(latencies)*1000:.2f}ms")
        dprint(f"中位延迟:           {median(latencies)*1000:.2f}ms")
        dprint(f"最小延迟:           {min(latencies)*1000:.2f}ms")
        dprint(f"最大延迟:           {max(latencies)*1000:.2f}ms")

        sorted_latencies = sorted(latencies)
        p95_idx = int(len(sorted_latencies) * 0.95)
        p99_idx = int(len(sorted_latencies) * 0.99)
        if p95_idx < len(sorted_latencies):
            dprint(f"95%延迟:            {sorted_latencies[p95_idx]*1000:.2f}ms")
        if p99_idx < len(sorted_latencies):
            dprint(f"99%延迟:            {sorted_latencies[p99_idx]*1000:.2f}ms")

    dprint("=" * 60)


@dataclass
class SOSClientConfig:
    url: str = "http://localhost:8000/v1/chat/completions"
    timeout: int = 30


@dataclass
class RequestResult:
    identity: str
    success: bool
    error: Optional[str]
    response: Optional[str]
    probs: Optional[dict]
    latency: float


class SOSClient:
    def __init__(self, config: SOSClientConfig=SOSClientConfig()):
        self.config = config

    async def send_req(
        self, session: aiohttp.ClientSession, audio
    ) -> RequestResult:
        payload = copy.deepcopy(PAYLOAD)
        payload["messages"][0]["content"][1]["audio_url"]["url"] = audio[1]

        start_time = time.perf_counter()
        try:
            async with session.post(
                self.config.url,
                json=payload,
                headers=HEADERS,
                timeout=aiohttp.ClientTimeout(total=self.config.timeout),
            ) as response:
                response_text = await response.text()
                latency = time.perf_counter() - start_time
                delay = (int)(latency * 1000)
                # dprint(f"{audio[0]}: delay {delay}ms")
                self.latencies.append(delay)
                if len(self.latencies) % 10 == 0:
                    dprint(f"<STATISTICS> process {os.getpid()} average latency {int(np.mean(self.latencies))}ms\n")

                if response.status == 200:
                    return RequestResult(
                        identity=audio[0],
                        success=True,
                        error=None,
                        response=response_text,
                        probs=response_to_probs(response_text),
                        latency=latency,
                    )
                else:
                    dprint(f"HTTP error {response.status}")
                    assert False
                    return RequestResult(
                        identity=audio[0],
                        success=False,
                        error=f"HTTP {response.status}: {response_text}",
                        response=None,
                        latency=latency,
                    )

        except Exception as e:
            dprint(f"exception {str(e)}")
            assert False
            latency = time.perf_counter() - start_time
            return RequestResult(
                identity=audio[0],
                success=False,
                error=str(e),
                response=None,
                latency=latency,
            )

    async def worker(
        self, session: aiohttp.ClientSession, audios: list
    ) -> List[RequestResult]:
        request_results = []

        # for idx in range(len(audios)):
        for idx in tqdm(range(len(audios))):
            result = await self.send_req(session, audios[idx])
            request_results.append(result)

        return request_results

    async def __call__(self, audios, concurrence=1) -> bool:
        self.latencies = []
        input_audios = distribute_audios(audios, concurrence)
        results = []

        connector = aiohttp.TCPConnector(
            limit=concurrence * 2,
            limit_per_host=concurrence,
            ttl_dns_cache=300,
            use_dns_cache=True,
            enable_cleanup_closed=True,
        )
        timeout = aiohttp.ClientTimeout(
            total=self.config.timeout, connect=10, sock_read=self.config.timeout
        )

        start_time = time.perf_counter()
        async with aiohttp.ClientSession(
            connector=connector, timeout=timeout, connector_owner=True
        ) as session:
            tasks = []
            for worker_id in range(concurrence):
                task = asyncio.create_task(
                    self.worker(session, input_audios[worker_id])
                )
                tasks.append(task)
            dprint("⏳ 执行测试中...")
            worker_results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = time.perf_counter()

        for worker_result in worker_results:
            if isinstance(worker_result, Exception):
                dprint(f"exception in results")
                assert False
                continue
            results.extend(worker_result)
        assert len(results) == len(audios)

        print_results(results, concurrence, start_time, end_time)
        return results
