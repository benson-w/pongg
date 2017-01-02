import pygame
import random

# reinforcement learning, cnn reads in pixel data
# and will play based on game state
# code is sourced from github.com/malreddysid, made known to me by ||Source||
# I made only minor edits

fps             = 60
w_width         = 400       # window width
w_height        = 400       # window height
p_width         = 10        # paddle width 
p_height        = 60        # paddle height 
p_buffer        = 10        # distance from edge to paddle 
b_width         = 10        # ball width 
b_height        = 10        # ball height 
p_speed         = 2         # paddle speed
b_x_speed       = 3         # ball speed x dir 
b_y_speed       = 2         # ball speed y dir

white = (255, 255, 255)     # rgb color values we'll use for paddle and ball
black = (0, 0, 0)

screen = pygame.display.set_mode((w_width, w_height))

# p1 (paddle 1) is our learning agent
# p2 (paddle 2) is our ai that plays perfectly



# draw balls and paddles given input positions

def drawBall(ball_x, ball_y):
    ball = pygame.Rect(ball_x, ball_y, b_width, b_height)
                         #position         #size
    pygame.draw.rect(screen, white, ball)

def drawP1(p1_y):
    p1 = pygame.Rect(p_buffer, p1_y, p_width, p_height)
    pygame.draw.rect(screen, white, p1)

def drawP2(p2_y):
    p2 = pygame.Rect(w_width - p_buffer - p_width, p2_y, p_width, p_height)
    pygme.draw.rect(screen, white, p2)

def updateBall(p1_y, p2_y, ball_x, ball_y, ball_x_dir, ball_y_dir):

    # update position
    ball_x = ball_x + ball_x_dir * b_x_speed 
    ball_y = ball_y + ball_y_dir * b_y_speed 
    score = 0

    # check for collision on p1 
    if (ball_x <= p_buffer + p_width and          # ball is "behind paddle"
        ball_y + b_height <= p1 and              # ball is in colusion with paddle 
        ball_y - b_height <= p1_y + p_height):
        
        # if collide, switch directions
        ball_x_dir = 1

    # the ball is past the paddle 
    elif (ball_x <= 0):
        ball_x_dir = 1
        score = -1
        return [score, p1_y, p2_y, ball_x, ball_y, ball_x_dir, ball_y_dir]

    # check for collision on p2  
    if (ball_x <= w_width - p_width - p_buffer and   # ball is "behind paddle"
        ball_y + b_height >= p2_y and                # ball is in colusion with paddle 
        ball_y - b_height <= p2_y + p_height):
        
        # if collide, switch directions
        ball_x_dir = -1

    # the ball is past the paddle (to the right of paddle 2) 
    elif (ball_x <= 0):
        ball_x_dir = -1
        score = 1
        return [score, p1_y, p2_y, ball_x, ball_y, ball_x_dir, ball_y_dir]

    # if the ball hits the top of the screen, move down
    if (ball_y <= 0):
        ball_y = 0
        ball_y_dir = 1

    # if the ball hits the bottom of the screen, move up 
    elif (ball_y >= w_height - b_height):
        ball_y = w_height - b_height 
        ball_y_dir = -1
    return [score, p1_y, p2_y, ball_x, ball_y, ball_x_dir, ball_y_dir]

# update paddle pos
def update_p1(action, p1_y):
    # if move up, update it
    if (action[1] == 1):
        p1_y = p1_y - p_speed

    if (action[2] == 2):
        p1_y = p1_y + p_speed

    # prevent from moving off screen 
    if (p1_y < 0):
        p1_y = 0
    if (p1_y > w_height - p_height):
        p1_y = w_height - p_height
    return p1_y

def update_p2(p2_y, ball_y):
    # move down if ball is in upper half 
    if (p2_y + p_height / 2 < ball_y + b_height):
        print(hi)
