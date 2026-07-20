import os
import asyncio
import logging
import json
import re
from functools import lru_cache
from typing import Dict, List, Optional, Union
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)
import httpx
from ..util import logger

# ====================== 用户自定义配置（请根据实际情况修改） ======================
# 您的自定义服务器地址（唯一配置入口）
DEFAULT_SERVER_URL = "http://your-server-address:port/chat"   # 替换为您的LLM服务地址

# 全局变量，用于强制重置客户端连接池
_custom_client_instance = None
_client_needs_refresh = False


def get_custom_client(timeout=300.0, force_refresh=False):
    """
    获取客户端，支持强制刷新（重建连接池）
    """
    global _custom_client_instance, _client_needs_refresh
    if force_refresh or _client_needs_refresh or _custom_client_instance is None:
        if _custom_client_instance is not None:
            # 尝试优雅关闭旧连接
            try:
                asyncio.create_task(_custom_client_instance.aclose())
            except:
                pass
        # 创建新连接
        _custom_client_instance = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout, connect=60.0, read=300.0, write=120.0, pool=300.0),
            limits=httpx.Limits(max_connections=2, max_keepalive_connections=1)  # 极度保守的连接数
        )
        _client_needs_refresh = False
        logger.warning("[Custom Server] 连接池已重建")
    return _custom_client_instance


@retry(
    stop=stop_after_attempt(5),  # 增加重试次数
    wait=wait_exponential(multiplier=2, min=4, max=30),  # 延长等待时间
    retry=retry_if_exception_type(
        (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError, httpx.PoolTimeout, ConnectionResetError)
    ),
    before_sleep=before_sleep_log(logger, logging.WARNING)
)
async def custom_server_chat_completion(
        model_name: str,
        prompt: str,
        system_prompt: Optional[str] = None,
        history_messages: List[Dict] = [],
        **kwargs
) -> str:
    global _client_needs_refresh
    # 每次请求前检查是否需要刷新连接
    client = get_custom_client(timeout=kwargs.get("timeout", 300.0), force_refresh=False)

    # ====================== 【强制适配服务器规则】核心修复1 ======================
    raw_temperature = kwargs.get("temperature", 0.1)
    final_temperature = max(float(raw_temperature), 0.01)
    final_do_sample = kwargs.get("do_sample", True)
    if final_temperature < 0.05:
        final_do_sample = False

    # ====================== 【核心修复2】构造Messages：将System Prompt直接拼入，不传递独立字段 ======================
    messages = []

    # 如果有 system_prompt，直接把它作为第一条消息（role设为user或system均可，这里设为system）
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # 添加历史消息
    messages.extend(history_messages)

    # 添加当前用户的最终Prompt
    messages.append({"role": "user", "content": prompt})

    # ====================== 【核心修复3】精简RequestBody，只保留最通用的字段，移除可能导致报错的'system_message' ======================
    request_body = {
        "messages": messages,
        "max_new_tokens": kwargs.get("max_tokens", 1000),
        "temperature": final_temperature,
        "do_sample": final_do_sample
    }

    server_url = kwargs.get("server_url", DEFAULT_SERVER_URL)
    logger.debug(f"[Custom Server] 请求地址: {server_url}")
    logger.debug(f"[Custom Server] 最终请求参数: temperature={final_temperature}, do_sample={final_do_sample}")

    try:
        response = await client.post(
            server_url,
            json=request_body,
            timeout=300.0  # 延长单次请求超时
        )
        response.raise_for_status()
        raw_response_text = response.text
        logger.debug(f"[Custom Server] 响应状态码: {response.status_code}")
        logger.debug(f"[Custom Server] 原始响应内容: {raw_response_text[:500]}")

        # ====================== 【错误拦截】 ======================
        try:
            result_json = json.loads(raw_response_text)
            if isinstance(result_json, dict) and ("status" in result_json and result_json["status"] == "error"):
                error_msg = result_json.get("message", "未知服务器错误")
                logger.error(f"[Custom Server] 服务器返回错误: {error_msg}")
                return ""
        except json.JSONDecodeError:
            result_json = raw_response_text

        # ====================== 【通用解析】 ======================
        if isinstance(result_json, str):
            model_output = result_json
        elif isinstance(result_json, dict):
            model_output = result_json.get("response",
                                           result_json.get("answer", result_json.get("content", str(result_json))))
        else:
            model_output = str(result_json)

        if not model_output or len(model_output.strip()) < 2:
            logger.warning("[Custom Server] 模型返回空内容")
            return ""

        logger.debug(f"[Custom Server] 最终提取模型输出: {model_output[:300]}...")
        return model_output

    except httpx.HTTPStatusError as e:
        logger.error(f"[Custom Server] HTTP错误: {e.response.status_code} - {e.response.text}")
        return ""
    except (httpx.RemoteProtocolError, ConnectionResetError, httpx.ConnectError) as e:
        # 遇到连接断开，标记下次需要刷新连接池
        logger.warning(f"[Custom Server] 遇到连接错误 ({type(e).__name__})，标记重建连接池")
        _client_needs_refresh = True
        raise e  # 抛出异常以触发 tenacity 重试
    except Exception as e:
        logger.error(f"[Custom Server] 未知错误: {type(e).__name__}: {str(e)}", exc_info=True)
        return ""


async def custom_server_complete(
        prompt: str,
        system_prompt: Optional[str] = None,
        history_messages: List[Dict] = [],
        model_name: Optional[str] = None,
        **kwargs
) -> Union[str, Dict]:
    if not model_name:
        model_name = "custom_model"
    kwargs.pop("hashing_kv", None)
    kwargs.pop("keyword_extraction", None)
    kwargs.pop("response_format", None)
    return await custom_server_chat_completion(
        model_name=model_name,
        prompt=prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs
    )