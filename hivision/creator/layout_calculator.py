#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
@DATE: 2024/9/5 21:35
@File: layout_calculator.py
@IDE: pycharm
@Description:
    布局计算器
"""

import cv2.detail
import numpy as np


def judge_layout(
    input_width,
    input_height,
    PHOTO_INTERVAL_W,
    PHOTO_INTERVAL_H,
    LIMIT_BLOCK_W,
    LIMIT_BLOCK_H,
):
    centerBlockHeight_1, centerBlockWidth_1 = (
        input_height,
        input_width,
    )  # 由证件照们组成的一个中心区块（1 代表不转置排列）
    centerBlockHeight_2, centerBlockWidth_2 = (
        input_width,
        input_height,
    )  # 由证件照们组成的一个中心区块（2 代表转置排列）

    # 1.不转置排列的情况下：
    layout_col_no_transpose = 0  # 行
    layout_row_no_transpose = 0  # 列
    for i in range(1, 4):  # 最大3行
        centerBlockHeight_temp = input_height * i + PHOTO_INTERVAL_H * (i - 1)
        if centerBlockHeight_temp < LIMIT_BLOCK_H:
            centerBlockHeight_1 = centerBlockHeight_temp
            layout_row_no_transpose = i
        else:
            break
    for j in range(1, 4):  # 最大3列，限制为3x3
        centerBlockWidth_temp = input_width * j + PHOTO_INTERVAL_W * (j - 1)
        if centerBlockWidth_temp < LIMIT_BLOCK_W:
            centerBlockWidth_1 = centerBlockWidth_temp
            layout_col_no_transpose = j
        else:
            break
    layout_number_no_transpose = layout_row_no_transpose * layout_col_no_transpose

    # 2.转置排列的情况下：
    layout_col_transpose = 0  # 行
    layout_row_transpose = 0  # 列
    for i in range(1, 4):  # 最大3行
        centerBlockHeight_temp = input_width * i + PHOTO_INTERVAL_H * (i - 1)
        if centerBlockHeight_temp < LIMIT_BLOCK_H:
            centerBlockHeight_2 = centerBlockHeight_temp
            layout_row_transpose = i
        else:
            break
    for j in range(1, 4):  # 最大3列，限制为3x3
        centerBlockWidth_temp = input_height * j + PHOTO_INTERVAL_W * (j - 1)
        if centerBlockWidth_temp < LIMIT_BLOCK_W:
            centerBlockWidth_2 = centerBlockWidth_temp
            layout_col_transpose = j
        else:
            break
    layout_number_transpose = layout_row_transpose * layout_col_transpose

    if layout_number_transpose > layout_number_no_transpose:
        layout_mode = (layout_col_transpose, layout_row_transpose, 2)
        return layout_mode, centerBlockWidth_2, centerBlockHeight_2
    else:
        layout_mode = (layout_col_no_transpose, layout_row_no_transpose, 1)
        return layout_mode, centerBlockWidth_1, centerBlockHeight_1


def generate_layout_array(input_height, input_width, LAYOUT_WIDTH=1795, LAYOUT_HEIGHT=1205):
    # 1.基础参数表
    PHOTO_INTERVAL_H = 0  # 证件照与证件照之间的垂直距离
    PHOTO_INTERVAL_W = 0  # 证件照与证件照之间的水平距离
    SIDES_INTERVAL_H = 70  # 证件照与画布边缘的垂直距离
    SIDES_INTERVAL_W = 70  # 证件照与画布边缘的水平距离
    LIMIT_BLOCK_W = LAYOUT_WIDTH - 2 * SIDES_INTERVAL_W
    LIMIT_BLOCK_H = LAYOUT_HEIGHT - 2 * SIDES_INTERVAL_H

    # 2.创建一个 1180x1746 的空白画布
    white_background = np.zeros([LAYOUT_HEIGHT, LAYOUT_WIDTH, 3], np.uint8)
    white_background.fill(255)

    # 3.计算照片的 layout（列、行、横竖朝向）,证件照组成的中心区块的分辨率
    layout_mode, centerBlockWidth, centerBlockHeight = judge_layout(
        input_width,
        input_height,
        PHOTO_INTERVAL_W,
        PHOTO_INTERVAL_H,
        LIMIT_BLOCK_W,
        LIMIT_BLOCK_H,
    )
    # 4.开始排列组合
    x11 = (LAYOUT_WIDTH - centerBlockWidth) // 2
    y11 = (LAYOUT_HEIGHT - centerBlockHeight) // 2
    typography_arr = []
    typography_rotate = False
    if layout_mode[2] == 2:
        input_height, input_width = input_width, input_height
        typography_rotate = True

    for j in range(layout_mode[1]):
        for i in range(layout_mode[0]):
            xi = x11 + i * input_width + i * PHOTO_INTERVAL_W
            yi = y11 + j * input_height + j * PHOTO_INTERVAL_H
            typography_arr.append([xi, yi])

    return typography_arr, typography_rotate


def generate_layout_image(
    input_image, typography_arr, typography_rotate, width=295, height=413, 
    crop_line:bool=False,
    LAYOUT_WIDTH=1795, 
    LAYOUT_HEIGHT=1205,
):
  
    # 创建一个白色背景的空白画布
    white_background = np.zeros([LAYOUT_HEIGHT, LAYOUT_WIDTH, 3], np.uint8)
    white_background.fill(255)
    
    # 如果输入图像的高度不等于指定高度，则调整图像大小
    if input_image.shape[0] != height:
        input_image = cv2.resize(input_image, (width, height))
    
    # 如果需要旋转排版，则对图像进行转置和垂直镜像
    if typography_rotate:
        input_image = cv2.transpose(input_image)
        input_image = cv2.flip(input_image, 0)  # 0 表示垂直镜像

        # 交换高度和宽度
        height, width = width, height
    
    # 将图像按照排版数组中的位置放置到白色背景上
    for arr in typography_arr:
        locate_x, locate_y = arr[0], arr[1]
        white_background[locate_y : locate_y + height, locate_x : locate_x + width] = (
            input_image
        )

    if crop_line:
        # 添加裁剪线（只在照片外围显示）
        line_color = (100, 100, 100)  # 深灰色
        line_thickness = 1

        # 收集所有照片的位置信息
        photo_positions = []
        for arr in typography_arr:
            x, y = arr[0], arr[1]
            photo_positions.append((x, y, x + width, y + height))

        # 获取所有垂直和水平边界线
        vertical_lines = set()
        horizontal_lines = set()
        for x, y, x2, y2 in photo_positions:
            vertical_lines.add(x)
            vertical_lines.add(x2)
            horizontal_lines.add(y)
            horizontal_lines.add(y2)

        # 绘制垂直裁剪线（避开照片内部）
        for x in sorted(vertical_lines):
            # 找到该垂直线上被照片占用的区间
            occupied_ranges = []
            for px, py, px2, py2 in photo_positions:
                if px <= x <= px2:
                    occupied_ranges.append((py, py2))
            
            # 合并重叠区间
            occupied_ranges.sort()
            merged_ranges = []
            for start, end in occupied_ranges:
                if merged_ranges and start <= merged_ranges[-1][1]:
                    merged_ranges[-1] = (merged_ranges[-1][0], max(merged_ranges[-1][1], end))
                else:
                    merged_ranges.append((start, end))
            
            # 绘制非占用区域的线段
            current_y = 0
            for start, end in merged_ranges:
                if current_y < start:
                    cv2.line(white_background, (x, current_y), (x, start), line_color, line_thickness)
                current_y = end
            if current_y < LAYOUT_HEIGHT:
                cv2.line(white_background, (x, current_y), (x, LAYOUT_HEIGHT), line_color, line_thickness)

        # 绘制水平裁剪线（避开照片内部）
        for y in sorted(horizontal_lines):
            # 找到该水平线上被照片占用的区间
            occupied_ranges = []
            for px, py, px2, py2 in photo_positions:
                if py <= y <= py2:
                    occupied_ranges.append((px, px2))
            
            # 合并重叠区间
            occupied_ranges.sort()
            merged_ranges = []
            for start, end in occupied_ranges:
                if merged_ranges and start <= merged_ranges[-1][1]:
                    merged_ranges[-1] = (merged_ranges[-1][0], max(merged_ranges[-1][1], end))
                else:
                    merged_ranges.append((start, end))
            
            # 绘制非占用区域的线段
            current_x = 0
            for start, end in merged_ranges:
                if current_x < start:
                    cv2.line(white_background, (current_x, y), (start, y), line_color, line_thickness)
                current_x = end
            if current_x < LAYOUT_WIDTH:
                cv2.line(white_background, (current_x, y), (LAYOUT_WIDTH, y), line_color, line_thickness)

    # 返回排版后的图像
    return white_background
