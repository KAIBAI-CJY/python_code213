import pygame
import random

# 初始化游戏
pygame.init()

# 游戏窗口大小
window_width = 800
window_height = 600

# 颜色定义
white = (255, 255, 255)
black = (0, 0, 0)
red = (255, 0, 0)

# 创建游戏窗口
window = pygame.display.set_mode((window_width, window_height))
pygame.display.set_caption('贪吃蛇')

# 游戏时钟
clock = pygame.time.Clock()

# 蛇的初始位置和大小
snake_block_size = 20
snake_speed = 15

font_style = pygame.font.SysFont(None, 50)
score_font = pygame.font.SysFont(None, 35)

def our_snake(snake_block_size, snake_list):
    for x in snake_list:
        pygame.draw.rect(window, black, [x[0], x[1], snake_block_size, snake_block_size])

def message(msg, color):
    mesg = font_style.render(msg, True, color)
    window.blit(mesg, [window_width / 6, window_height / 3])

def game_loop():
    game_over = False
    game_close = False

    # 蛇的初始位置
    x1 = window_width / 2
    y1 = window_height / 2

    # 蛇的初始移动方向
    x1_change = 0
    y1_change = 0

    # 蛇的初始长度
    snake_list = []
    length_of_snake = 1

    # 食物的初始位置
    foodx = round(random.randrange(0, window_width - snake_block_size) / 20) * 20
    foody = round(random.randrange(0, window_height - snake_block_size) / 20) * 20

    while not game_over:

        while game_close == True:
            window.fill(white)
            message("游戏结束！按 Q-退出或 C-重新开始", red)
            pygame.display.update()

            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        game_over = True
                        game_close = False
                    if event.key == pygame.K_c:
                        game_loop()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game_over = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    x1_change = -snake_block_size
                    y1_change = 0
                elif event.key == pygame.K_RIGHT:
                    x1_change = snake_block_size
                    y1_change = 0
                elif event.key == pygame.K_UP:
                    y1_change = -snake_block_size
                    x1_change = 0
                elif event.key == pygame.K_DOWN:
                    y1_change = snake_block_size
                    x1_change = 0

        if x1 >= window_width or x1 < 0 or y1 >= window_height or y1 < 0:
            game_close = True
        x1 += x1_change
        y1 += y1_change
        window.fill(white)
        pygame.draw.rect(window, red, [foodx, foody, snake_block_size, snake_block_size])
        snake_head = []
        snake_head.append(x1)
        snake_head.append(y1)
        snake_list.append(snake_head)
        if len(snake_list) > length_of_snake:
            del snake_list[0]

        for x in snake_list[:-1]:
            if x == snake_head:
                game_close = True

        our_snake(snake_block_size, snake_list)

        pygame.display.update()

        if x1 == foodx and y1 == foody:
            foodx = round(random.randrange(0, window_width - snake_block_size) / 20) * 20
            foody = round(random.randrange(0, window_height - snake_block_size) / 20) * 20
            length_of_snake += 1

        clock.tick(snake_speed)

    pygame.quit()

game_loop()
