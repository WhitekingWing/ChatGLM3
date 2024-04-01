#!/usr/bin/env python
# -*- coding: utf-8 -*-
import streamlit as st
import gymnasium as gym
import envScenario

from pathlib import Path
from typing import Annotated, Union

import typer
from peft import AutoPeftModelForCausalLM, PeftModelForCausalLM
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

ModelType = Union[PreTrainedModel, PeftModelForCausalLM]
TokenizerType = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]

app = typer.Typer(pretty_exceptions_show_locals=False)


def _resolve_path(path: Union[str, Path]) -> Path:
    return Path(path).expanduser().resolve()

# 使用 Streamlit 缓存装饰器
@st.cache_resource
def load_model_and_tokenizer(model_dir: Union[str, Path]) -> tuple[ModelType, TokenizerType]:
    model_dir = _resolve_path(model_dir)
    if (model_dir / 'adapter_config.json').exists():
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_dir, trust_remote_code=True, device_map='auto'
        )
        tokenizer_dir = model.peft_config['default'].base_model_name_or_path
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, trust_remote_code=True, device_map='auto'
        )
        tokenizer_dir = model_dir
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_dir, trust_remote_code=True
    )
    return model, tokenizer
@app.command()
def main(
        #model_dir: Annotated[str, typer.Argument(help='')],
        #prompt: Annotated[str, typer.Option(help='')],
):
    envType = 'highway-v0'
    env = gym.make(envType, render_mode="rgb_array")
    st.title("ChatGLM 司机")
    # 默认的 model_dir 和 prompt
    default_model_dir = r"./output/checkpoint-2000/"
    default_prompt = "您正在一条有5条车道的路上行驶，目前您在最右侧车道。您的当前位置是(322.81, 16.00)，速度为25.00米/秒，加速度为0.00米/秒^2，车道位置为322.81米。周围还有其他车辆在行驶，以下是它们的基本信息：车辆464与您在同一车道，位于您前方。它的位置是(350.29, 16.00)，速度为25.00米/秒，加速度为0.00米/秒^2，车道位置为350.29米。 车辆208在您左侧的车道行驶，也位于您前方。它的位置是(383.35, 12.00)，速度为25.00米/秒，加速度为0.00米/秒^2，车道位置为383.35米。"
    model, tokenizer = load_model_and_tokenizer(default_model_dir)
    # 用户输入
    #model_dir = st.text_input("请输入模型目录的路径", default_model_dir)
    #prompt = st.text_area("请输入提示词", default_prompt)
    confirm_button = st.button("开始")
    if confirm_button:
        i = 0
        done = truncated = False
        obs, info = env.reset()
        sce = envScenario.EnvScenario(env,envType,3)
        while not (done or truncated):
            img = env.render()
            # 使用 Streamlit 显示图像
            st.image(img)
            prompt = sce.describe(4)
            if prompt:
                st.text_area("道路环境:", value=prompt, height=300,key=f'text_area_{i}')
                i+=1
                response, _ = model.chat(tokenizer, str(prompt))
                st.text_area("助手:", value=response, height=300,key=f'text_area_{i}')
                i+=1
            else:
                st.error("请输入提示词。")
            if response[-1] == '1':
                action = env.action_type.actions_indexes["IDLE"]
            elif response[-1] == '2':
                action = env.action_type.actions_indexes["LANE_RIGHT"]
            elif response[-1] == '3':
                action = env.action_type.actions_indexes["FASTER"]
            elif response[-1] == '0':
                action = env.action_type.actions_indexes["LANE_LEFT"]
            elif response[-1] == '4':
                action = env.action_type.actions_indexes["SLOWER"]
            else:
                action = action = env.action_type.actions_indexes["IDLE"]
            obs, reward, done, truncated, info = env.step(action)



if __name__ == '__main__':
    app()
