import glfw
import numpy as np
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import pyrr
from PIL import Image

# SHADERS GLSL

vertex_src = """
//# version 330

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 color;
layout(location = 2) in vec2 texCoord;

uniform mat4 model;

out vec3 v_color;
out vec2 out_texCoord;

void main()
{
    gl_Position = model * vec4(position, 1.0);
    v_color = color;
    out_texCoord = texCoord;
}
"""

fragment_src = """
//# version 330

in vec3 v_color;
in vec2 out_texCoord;

uniform sampler2D texture1;

out vec4 out_color;

void main()
{
    float amb = 1.0;
    out_color = texture(texture1, out_texCoord);
}
"""

# window init
glfw.init()
window = glfw.create_window(800, 600, "Act window", None, None)

if not window:
    glfw.terminate()
    exit()

glfw.make_context_current(window)

# OBJECT CREATION
# VERTEX DEFINITION
vertices = [
    -0.5, 1, 0, 1.0, 0.0, 0.0, 0, 0,
    -1, 0, 0, 0.0, 1.0, 0.0, 0, 1,
    0, 0, 0, 0.0, 0.0, 1.0, 1, 1,
    
    1, 1, 0, 1.0, 1.0, 0.0, 1, 0,
    0, 0, 0, 0.0, 1.0, 1.0, 0, 1,
    1, 0, 0, 1.0, 0.0, 1.0, 1, 1,
    
    -1, 0, 0, 1.0, 0.0, 0.0, 1.5, 0,
    -1, -1, 0, 0.0, 1.0, 0.0, 1.5, 1.5,
    0, -1, 0, 0.0, 0.0, 1.0, 0, 1.5,
    0, 0, 0, 1.0, 1.0, 0.0, 0, 0,
    
    0, 0, 0, 1.0, 1.0, 0.0, 0, 0,
    0.5, -1, 0, 0.0, 1.0, 1.0, 0, 1.5,
    1, -1, 0, 1.0, 0.0, 1.0, 1.5, 1.5,
    0.5, 0, 1.0, 1.0, 1.0, 1.0, 1.5, 0
]


indices = [
    0, 1, 2,
    3, 4, 5,
    6, 7, 8, 6, 8, 9,
    10, 11, 12, 10, 12, 13
]

vertices = np.array(vertices, dtype=np.float32)
indices = np.array(indices, dtype=np.uint32)


# SENDING DATA
# ================================================================
# vao
vao = glGenVertexArrays(1)
glBindVertexArray(vao)

# VBO
vbo = glGenBuffers(1)
glBindBuffer(GL_ARRAY_BUFFER, vbo)
glBufferData(GL_ARRAY_BUFFER, vertices.nbytes,
             vertices, GL_STATIC_DRAW)

# vertex
glEnableVertexAttribArray(0)
glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(0))

# color
glEnableVertexAttribArray(1)
glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(12))

# Textcoord
glEnableVertexAttribArray(2)
glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(24))
# indexing
ebo = glGenBuffers(1)
glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes,
             indices, GL_STATIC_DRAW)
glBindVertexArray(0)

# IMg

img = Image.open("GVC-Acts/Finals/crate-texture_mod.png")
img_data = np.array(list(img.getdata()), np.uint8)
print(img_data)
texture = glGenTextures(1)
glBindTexture(GL_TEXTURE_2D, texture)
glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, img.width, img.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, img_data)  
glGenerateMipmap(GL_TEXTURE_2D)
# =============================================================
# Transformation
scale = pyrr.Matrix44.from_scale(pyrr.Vector3([1, 1, 1]))

# # Translation

# SHADER SPACE
shader = compileProgram(compileShader(
    vertex_src, GL_VERTEX_SHADER), compileShader(fragment_src, GL_FRAGMENT_SHADER))
glUseProgram(shader)

model = scale

# model_loc = glGetUniformLocation(shader, "model")
# glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)

model_loc = glGetUniformLocation(shader, "model")
tex_loc = glGetUniformLocation(shader, "texture1")
glUniform1i(tex_loc, 0)

glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)

# RENDERING SPACE
glClearColor(0.1, 0.1, 0.1, 1)
glEnable(GL_DEPTH_TEST)  # activate the z buffer
while not glfw.window_should_close(window):
    glfw.poll_events()
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    # # # OBJECT TRANSFORMATION
    # roty = pyrr.matrix44.create_from_y_rotation(
    #     np.radians(20 * glfw.get_time()))
    # model = roty
    # # model = pyrr.matrix44.multiply(model, translation)
    # glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)

    # OBJECT ASSEMBLY AND RENDERING
    glBindVertexArray(vao)
    glDrawElements(GL_TRIANGLES, len(indices), GL_UNSIGNED_INT, None)
    glBindVertexArray(0)

    glfw.swap_buffers(window)
# MEMORY CLEARING
glDeleteBuffers(2, [vbo, ebo,])
glDeleteVertexArrays(1, [vao,])
glfw.terminate()
