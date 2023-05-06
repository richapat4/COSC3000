import glfw
import glfw.GLFW as GLFW_CONSTANTS
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram,compileShader
import numpy as np
import pyrr
import ctypes
from PIL import Image, ImageOps

############################## Constants ######################################

SCREEN_WIDTH = 640
SCREEN_HEIGHT = 480

RETURN_ACTION_CONTINUE = 0
RETURN_ACTION_EXIT = 1

#0: debug, 1: production
GAME_MODE = 0
LIGHT_COUNT = 4

############################## helper functions ###############################

def initialize_glfw():

    glfw.init()
    glfw.window_hint(GLFW_CONSTANTS.GLFW_CONTEXT_VERSION_MAJOR,3)
    glfw.window_hint(GLFW_CONSTANTS.GLFW_CONTEXT_VERSION_MINOR,3)
    glfw.window_hint(GLFW_CONSTANTS.GLFW_OPENGL_PROFILE, GLFW_CONSTANTS.GLFW_OPENGL_CORE_PROFILE)
    glfw.window_hint(GLFW_CONSTANTS.GLFW_OPENGL_FORWARD_COMPAT, GLFW_CONSTANTS.GLFW_TRUE)
    #for uncapped framerate
    glfw.window_hint(GLFW_CONSTANTS.GLFW_DOUBLEBUFFER,GL_FALSE) 
    window = glfw.create_window(SCREEN_WIDTH, SCREEN_HEIGHT, "Title", None, None)
    glfw.make_context_current(window)
    
    #glViewport(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT)
    
    glEnable(GL_PROGRAM_POINT_SIZE)
    glClearColor(0.1, 0.1, 0.1, 1)

    return window

##################################### Model ###################################

class Cube:


    def __init__(self, position, eulers, eulerVelocity):

        self.position = np.array(position, dtype=np.float32)
        self.eulers = np.array(eulers, dtype=np.float32)
        self.eulerVelocity = np.array(eulerVelocity, dtype=np.float32)

class Light:


    def __init__(self, position, color):

        self.position = np.array(position, dtype=np.float32)
        self.color = np.array(color, dtype=np.float32)

class Player:


    def __init__(self, position, eulers):
        self.position = np.array(position,dtype=np.float32)
        self.eulers = np.array(eulers,dtype=np.float32)
        self.moveSpeed = 1
        self.global_up = np.array([0, 0, 1], dtype=np.float32)

    def move(self, direction, amount):
        walkDirection = (direction + self.eulers[1]) % 360
        self.position[0] += amount * self.moveSpeed * np.cos(np.radians(walkDirection),dtype=np.float32)
        self.position[1] += amount * self.moveSpeed * np.sin(np.radians(walkDirection),dtype=np.float32)

    def increment_direction(self, theta_increase, phi_increase):
        self.eulers[1] = (self.eulers[1] + theta_increase) % 360
        self.eulers[0] = min(max(self.eulers[0] + phi_increase,-89),89)

    def get_forwards(self):

        return np.array(
            [
                #x = cos(theta) * cos(phi)
                np.cos(
                    np.radians(
                        self.eulers[1]
                    ),dtype=np.float32
                ) * np.cos(
                    np.radians(
                        self.eulers[0]
                    ),dtype=np.float32
                ),

                #y = sin(theta) * cos(phi)
                np.sin(
                    np.radians(
                        self.eulers[1]
                    ),dtype=np.float32
                ) * np.cos(
                    np.radians(
                        self.eulers[0]
                    ),dtype=np.float32
                ),

                #x = sin(phi)
                np.sin(
                    np.radians(
                        self.eulers[0]
                    ),dtype=np.float32
                )
            ], dtype = np.float32
        )
    
    def get_up(self):

        forwards = self.get_forwards()
        right = np.cross(
            a = forwards,
            b = self.global_up
        )

        return np.cross(
            a = right,
            b = forwards,
        )

class Scene:


    def __init__(self):
        i = 0
        j = 0
        k = 0

        self.cubes = []

        for i in range(15):
            for j in range(3):
                if i < 10:
                    position = [-1, ((-1 + i * 0.25) + 2), (-1 + j * 0.5) + 1 + (k * 0.25)]
                elif i > 9 and j == 1:
                    position = [-1, ((-1 + i * 0.25) + 2), (-1 + j * 0.5) + 1 + (k * 0.25)]
                self.cubes.append(Cube(position, [0, 90, 0], [0, 0, 0]))
                    

        for i in range(15):
            for j in range(3):
                if i < 10:
                    position = [-1, ((5 + i * 0.25) + 2), (-1 + j * 0.5) + 1]
                elif i > 9 and j == 1:
                    position = [-1, ((5  - (k * 0.25)) + 2), (-1 + j * 0.5) + 1]
                    k+=1
                self.cubes.append(Cube(position, [0, 90, 0], [0, 0, 0]))
                    

        #     Cube(
        #         position = [-1,((-1 + i * 0.25) + 2),(-1 + j * 0.5)+1], #x,y,z
        #         eulers = [0,90,0],
        #         eulerVelocity = [0,0,0],
        #     )
            
        #     for i in range(10 + (10 if j == 1 and i == 9 else 0))
        #     for j in range(3) 

        # ]
        i = 0
        m = 0
        self.lights = []

        for i in range(LIGHT_COUNT):
            if(i < 2):
                position = [-2,3+i,0+i]
            else:
                position = [-2,6+i, 0 + m]
                m+=1
            self.lights.append(Light(position,color = [1, 1, 1]))
            # Light(
            #     color = [1, 1, 1]
            # )  

        self.player = Player(
            position = [-4, 0, 0],
            eulers = [0, 0, 0]
        )
    
    def update(self, rate: float) -> None:

        for cube in self.cubes:
            cube.eulers = np.mod(
                cube.eulers + cube.eulerVelocity * rate, 
                [360, 360, 360], 
                dtype=np.float32
            )

##################################### Control #################################

class App:


    def __init__(self, window):

        self.window = window

        self.lastTime = 0
        self.currentTime = 0
        self.numFrames = 0
        self.frameTime = 0

        self.scene = Scene()

        self.engine = Engine(self.scene)

        self.mainLoop()

    def mainLoop(self):
        running = True
        while (running):
            #check events
            if glfw.window_should_close(self.window) \
                or glfw.get_key(self.window, GLFW_CONSTANTS.GLFW_KEY_ESCAPE) == GLFW_CONSTANTS.GLFW_PRESS:
                running = False
            
            self.handleKeys()
            self.handleMouse()

            glfw.poll_events()

            self.scene.update(self.frameTime / 16.67)
            
            self.engine.draw(self.scene)

            #timing
            self.showFrameRate()

        self.quit()

    def handleKeys(self):
        
        if glfw.get_key(self.window, GLFW_CONSTANTS.GLFW_KEY_W) == GLFW_CONSTANTS.GLFW_PRESS:
            self.scene.player.move(0, 0.0025*self.frameTime)
            return
        if glfw.get_key(self.window, GLFW_CONSTANTS.GLFW_KEY_A) == GLFW_CONSTANTS.GLFW_PRESS:
            self.scene.player.move(90, 0.0025*self.frameTime)
            return
        if glfw.get_key(self.window, GLFW_CONSTANTS.GLFW_KEY_S) == GLFW_CONSTANTS.GLFW_PRESS:
            self.scene.player.move(180, 0.0025*self.frameTime)
            return
        if glfw.get_key(self.window, GLFW_CONSTANTS.GLFW_KEY_D) == GLFW_CONSTANTS.GLFW_PRESS:
            self.scene.player.move(-90, 0.0025*self.frameTime)
            return

    def handleMouse(self):

        (x,y) = glfw.get_cursor_pos(self.window)
        rate = self.frameTime / 16.67
        theta_increment = rate * ((SCREEN_WIDTH / 2) - x)
        phi_increment = rate * ((SCREEN_HEIGHT / 2) - y)
        self.scene.player.increment_direction(theta_increment, phi_increment)
        glfw.set_cursor_pos(self.window, SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2)

    def showFrameRate(self):

        self.currentTime = glfw.get_time()
        delta = self.currentTime - self.lastTime
        if (delta >= 1):
            framerate = max(1,int(self.numFrames/delta))
            glfw.set_window_title(self.window, f"Running at {framerate} fps.")
            self.lastTime = self.currentTime
            self.numFrames = -1
            self.frameTime = float(1000.0 / max(1,framerate))
        self.numFrames += 1
    
    def quit(self):

        self.engine.quit()

##################################### View ####################################

class Engine:


    def __init__(self, scene):

        #initialise opengl
        glClearColor(0.1, 0.1, 0.1, 1)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_CULL_FACE)
        glCullFace(GL_BACK)
        
        self.create_framebuffers()
        self.create_assets(scene)
        self.set_up_shaders()

    def create_assets(self,scene):
        #create assets
        self.wood_texture = Material("goldBrick", "png")
        self.cube_mesh = ObjMesh("models/solar.obj")
        #generate position buffer
        self.cubeTransforms = np.array([
            pyrr.matrix44.create_identity(dtype = np.float32)

            for i in range(len(scene.cubes))
        ], dtype=np.float32)
        glBindVertexArray(self.cube_mesh.vao)
        self.cubeTransformVBO = glGenBuffers(1)
        glBindBuffer(
            GL_ARRAY_BUFFER, 
            self.cubeTransformVBO
        )
        glBufferData(
            GL_ARRAY_BUFFER, 
            self.cubeTransforms.nbytes, 
            self.cubeTransforms, 
            GL_STATIC_DRAW
        )
        glEnableVertexAttribArray(5)
        glVertexAttribPointer(5, 4, GL_FLOAT, GL_FALSE, 64, ctypes.c_void_p(0))
        glEnableVertexAttribArray(6)
        glVertexAttribPointer(6, 4, GL_FLOAT, GL_FALSE, 64, ctypes.c_void_p(16))
        glEnableVertexAttribArray(7)
        glVertexAttribPointer(7, 4, GL_FLOAT, GL_FALSE, 64, ctypes.c_void_p(32))
        glEnableVertexAttribArray(8)
        glVertexAttribPointer(8, 4, GL_FLOAT, GL_FALSE, 64, ctypes.c_void_p(48))
        glVertexAttribDivisor(5,1)
        glVertexAttribDivisor(6,1)
        glVertexAttribDivisor(7,1)
        glVertexAttribDivisor(8,1)



    def set_up_shaders(self):
        self.shaderTextured = self.createShader(
            "shaders\\vertex.txt", 
            "shaders\\fragment.txt"
        )
        self.shaderColored = self.createShader(
            "shaders\\simple_3d_vertex.txt", 
            "shaders\\simple_3d_fragment.txt"
        )
        

        #ADDING BLOOM
        # self.bloom_blur_shader = self.createShader("shaders/simple_post_vertex.txt", "shaders/bloom_blur_fragment.txt")
        # glUseProgram(self.bloom_blur_shader)
        # glUniform1i(glGetUniformLocation(self.bloom_blur_shader, "material"), 0)
        # glUniform1i(glGetUniformLocation(self.bloom_blur_shader, "bright_material"), 1)

        # self.bloom_transfer_shader = self.createShader("shaders/simple_post_vertex.txt", "shaders/bloom_transfer_fragment.txt")
        # glUseProgram(self.bloom_transfer_shader)
        # glUniform1i(glGetUniformLocation(self.bloom_transfer_shader, "material"), 0)
        # glUniform1i(glGetUniformLocation(self.bloom_transfer_shader, "bright_material"), 1)

        # self.bloom_resolve_shader = self.createShader("shaders/simple_post_vertex.txt", "shaders/bloom_resolve_fragment.txt")
        # glUseProgram(self.bloom_resolve_shader)
        # glUniform1i(glGetUniformLocation(self.bloom_resolve_shader, "material"), 0)
        # glUniform1i(glGetUniformLocation(self.bloom_resolve_shader, "bright_material"), 1)


        projection_transform = pyrr.matrix44.create_perspective_projection(
            fovy = 45, aspect = 640/480, 
            near = 0.1, far = 40, dtype=np.float32
        )

        glUseProgram(self.shaderTextured)
        #get shader locations
        self.viewLocTextured = glGetUniformLocation(self.shaderTextured, "view")
        self.lightLocTextured = {

            "pos": [
                glGetUniformLocation(
                    self.shaderTextured,f"lightPos[{i}]"
                ) 
                for i in range(LIGHT_COUNT)
            ],

            "color": [
                glGetUniformLocation(
                    self.shaderTextured,f"lights[{i}].color"
                ) 
                for i in range(LIGHT_COUNT)
            ],

            "strength": [
                glGetUniformLocation(
                    self.shaderTextured,f"lights[{i}].strength"
                ) 
                for i in range(LIGHT_COUNT)
            ],

            "count": glGetUniformLocation(
                self.shaderTextured,f"lightCount"
            )
        }

        self.cameraLocTextured = glGetUniformLocation(self.shaderTextured, "viewPos")

        #set up uniforms
        glUniformMatrix4fv(
            glGetUniformLocation(
                self.shaderTextured,"projection"
            ),
            1,GL_FALSE,projection_transform
        )

        glUniform3fv(
            glGetUniformLocation(
                self.shaderTextured,"ambient"
            ), 
            1, np.array([0.1, 0.1, 0.1],dtype=np.float32)
        )

        glUniform1i(
            glGetUniformLocation(
                self.shaderTextured, "material.albedo"
            ), 0
        )

        glUniform1i(
            glGetUniformLocation(
                self.shaderTextured, "material.ao"
            ), 1
        )

        glUniform1i(
            glGetUniformLocation(
                self.shaderTextured, "material.normal"
            ), 2
        )

        glUniform1i(
            glGetUniformLocation(
                self.shaderTextured, "material.specular"
            ), 3
        )

    # #ADDED BRIGHT MATERIAL TO FRAGMENT
    #     glUniform1i(
    #         glGetUniformLocation(
    #             self.shaderTextured, "bright_material"
    #             ), 4
    #         )

        glUseProgram(self.shaderColored)
        #get shader locations
        self.viewLocUntextured = glGetUniformLocation(self.shaderColored, "view")
        self.modelLocUntextured = glGetUniformLocation(self.shaderColored, "model")
        self.colorLocUntextured = glGetUniformLocation(self.shaderColored, "color")

        glUniformMatrix4fv(
            glGetUniformLocation(
                self.shaderColored,"projection"
            ),1,GL_FALSE,projection_transform
        )

        #create assets
        self.light_mesh = UntexturedCubeMesh(
            l = 0.5,
            w = 0.5,
            h = 0.5
        )


    def create_framebuffers(self):
        self.fbos = []
        self.colorBuffers = []
        self.depthStencilBuffers = []
        for i in range(2):
            self.fbos.append(glGenFramebuffers(1))
            glBindFramebuffer(GL_FRAMEBUFFER, self.fbos[i])
        
            new_color_buffer_0 = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, new_color_buffer_0)
            glTexImage2D(
                GL_TEXTURE_2D, 0, GL_RGB, 
                SCREEN_WIDTH, SCREEN_HEIGHT,
                0, GL_RGB, GL_UNSIGNED_BYTE, None
            )
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glBindTexture(GL_TEXTURE_2D, 0)
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, 
                                    GL_TEXTURE_2D, new_color_buffer_0, 0)
            
            new_color_buffer_1 = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, new_color_buffer_1)
            glTexImage2D(
                GL_TEXTURE_2D, 0, GL_RGB, 
                SCREEN_WIDTH, SCREEN_HEIGHT,
                0, GL_RGB, GL_UNSIGNED_BYTE, None
            )
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glBindTexture(GL_TEXTURE_2D, 0)
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, 
                                    GL_TEXTURE_2D, new_color_buffer_1, 0)
            
            self.colorBuffers.append([new_color_buffer_0, new_color_buffer_1])
            
            self.depthStencilBuffers.append(glGenRenderbuffers(1))
            glBindRenderbuffer(GL_RENDERBUFFER, self.depthStencilBuffers[i])
            glRenderbufferStorage(
                GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, SCREEN_WIDTH, SCREEN_HEIGHT
            )
            glBindRenderbuffer(GL_RENDERBUFFER,0)
            glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, 
                                        GL_RENDERBUFFER, self.depthStencilBuffers[i])

            glBindFramebuffer(GL_FRAMEBUFFER, 0)
    

    def createShader(self, vertexFilepath, fragmentFilepath):

        with open(vertexFilepath,'r') as f:
            vertex_src = f.readlines()

        with open(fragmentFilepath,'r') as f:
            fragment_src = f.readlines()

        shader = compileProgram(compileShader(vertex_src, GL_VERTEX_SHADER),
                            compileShader(fragment_src, GL_FRAGMENT_SHADER))
    
        
        return shader

    def draw(self, scene):
        #refresh screen
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        view_transform = pyrr.matrix44.create_look_at(
            eye = scene.player.position,
            target = scene.player.position + scene.player.get_forwards(),
            up = scene.player.get_up(),
            dtype = np.float32
        )

        glUseProgram(self.shaderTextured)
        glUniformMatrix4fv(
            self.viewLocTextured, 1, GL_FALSE, view_transform
        )
        glUniform3fv(self.cameraLocTextured, 1, scene.player.position)
        #lights
        glUniform1f(self.lightLocTextured["count"], min(LIGHT_COUNT,max(0,len(scene.lights))))

        for i, light in enumerate(scene.lights):
            glUniform3fv(self.lightLocTextured["pos"][i], 1, light.position)
            glUniform3fv(self.lightLocTextured["color"][i], 1, light.color)
            glUniform1f(self.lightLocTextured["strength"][i], 1)
        
        for i, cube in enumerate(scene.cubes):
            model_transform = pyrr.matrix44.create_identity(dtype=np.float32)
            model_transform = pyrr.matrix44.multiply(
                m1=model_transform, 
                m2=pyrr.matrix44.create_from_eulers(
                    eulers=np.radians(cube.eulers), dtype=np.float32
                )
            )
            model_transform = pyrr.matrix44.multiply(
                m1=model_transform, 
                m2=pyrr.matrix44.create_from_translation(
                    vec=np.array(cube.position),dtype=np.float32
                )
            )
            self.cubeTransforms[i] = model_transform
        
        glBindVertexArray(self.cube_mesh.vao)
        glBindBuffer(
            GL_ARRAY_BUFFER, 
            self.cubeTransformVBO
        )
        glBufferData(GL_ARRAY_BUFFER, self.cubeTransforms.nbytes, self.cubeTransforms, GL_STATIC_DRAW)
        self.wood_texture.use()
        glDrawArraysInstanced(GL_TRIANGLES, 0, self.cube_mesh.vertex_count, len(scene.cubes))
        
        glUseProgram(self.shaderColored)
        
        glUniformMatrix4fv(self.viewLocUntextured, 1, GL_FALSE, view_transform)

        for light in scene.lights:
            model_transform = pyrr.matrix44.create_from_translation(
                vec=np.array(light.position),dtype=np.float32
            )
            glUniformMatrix4fv(self.modelLocUntextured, 1, GL_FALSE, model_transform)
            glUniform3fv(self.colorLocUntextured, 1, light.color)
            glBindVertexArray(self.light_mesh.vao)
            glDrawArrays(GL_TRIANGLES, 0, self.light_mesh.vertex_count)

        glFlush()

    def quit(self):
        self.cube_mesh.destroy()
        self.light_mesh.destroy()
        self.wood_texture.destroy()
        glDeleteBuffers(1, (self.cubeTransformVBO,))
        glDeleteProgram(self.shaderTextured)
        glDeleteProgram(self.shaderColored)

class Material:
    def __init__(self, filename, filetype):

        self.textures = []

        #albedo : 0
        self.textures.append(glGenTextures(1))
        glBindTexture(GL_TEXTURE_2D, self.textures[0])
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        with Image.open(f"gfx/{filename}_albedo.{filetype}", mode = "r") as img:
            image_width,image_height = img.size
            img = img.convert("RGBA")
            img_data = bytes(img.tobytes())
            glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA,image_width,image_height,0,GL_RGBA,GL_UNSIGNED_BYTE,img_data)
        glGenerateMipmap(GL_TEXTURE_2D)
        
        #ambient occlusion : 1
        self.textures.append(glGenTextures(1))
        glBindTexture(GL_TEXTURE_2D, self.textures[1])
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        with Image.open(f"gfx/{filename}_ao.{filetype}", mode = "r") as img:
            image_width,image_height = img.size
            img = img.convert("RGBA")
            img_data = bytes(img.tobytes())
            glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA,image_width,image_height,0,GL_RGBA,GL_UNSIGNED_BYTE,img_data)
        glGenerateMipmap(GL_TEXTURE_2D)

        #normal : 2
        self.textures.append(glGenTextures(1))
        glBindTexture(GL_TEXTURE_2D, self.textures[2])
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        with Image.open(f"gfx/{filename}_normal.{filetype}", mode = "r") as img:
            image_width,image_height = img.size
            img = img.convert("RGBA")
            img_data = bytes(img.tobytes())
            glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA,image_width,image_height,0,GL_RGBA,GL_UNSIGNED_BYTE,img_data)
        glGenerateMipmap(GL_TEXTURE_2D)

        #specular : 3
        self.textures.append(glGenTextures(1))
        glBindTexture(GL_TEXTURE_2D, self.textures[3])
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        with Image.open(f"gfx/{filename}_specular.{filetype}", mode = "r") as img:
            image_width,image_height = img.size
            img = img.convert("RGBA")
            img_data = bytes(img.tobytes())
            glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA,image_width,image_height,0,GL_RGBA,GL_UNSIGNED_BYTE,img_data)
        glGenerateMipmap(GL_TEXTURE_2D)

    def use(self):

        for i in range(len(self.textures)):
            glActiveTexture(GL_TEXTURE0 + i)
            glBindTexture(GL_TEXTURE_2D,self.textures[i])
    
    def destroy(self):
        glDeleteTextures(len(self.textures), self.textures)

class UntexturedCubeMesh:


    def __init__(self, l, w, h):
        # x, y, z
        self.vertices = (
                -l/2, -w/2, -h/2,
                 l/2, -w/2, -h/2,
                 l/2,  w/2, -h/2,

                 l/2,  w/2, -h/2,
                -l/2,  w/2, -h/2,
                -l/2, -w/2, -h/2,

                -l/2, -w/2,  h/2,
                 l/2, -w/2,  h/2,
                 l/2,  w/2,  h/2,

                 l/2,  w/2,  h/2,
                -l/2,  w/2,  h/2,
                -l/2, -w/2,  h/2,

                -l/2,  w/2,  h/2,
                -l/2,  w/2, -h/2,
                -l/2, -w/2, -h/2,

                -l/2, -w/2, -h/2,
                -l/2, -w/2,  h/2,
                -l/2,  w/2,  h/2,

                 l/2,  w/2,  h/2,
                 l/2,  w/2, -h/2,
                 l/2, -w/2, -h/2,

                 l/2, -w/2, -h/2,
                 l/2, -w/2,  h/2,
                 l/2,  w/2,  h/2,

                -l/2, -w/2, -h/2,
                 l/2, -w/2, -h/2,
                 l/2, -w/2,  h/2,

                 l/2, -w/2,  h/2,
                -l/2, -w/2,  h/2,
                -l/2, -w/2, -h/2,

                -l/2,  w/2, -h/2,
                 l/2,  w/2, -h/2,
                 l/2,  w/2,  h/2,

                 l/2,  w/2,  h/2,
                -l/2,  w/2,  h/2,
                -l/2,  w/2, -h/2
            )
        self.vertex_count = len(self.vertices)//3
        self.vertices = np.array(self.vertices, dtype=np.float32)

        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)

        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))

        # glEnableVertexAttribArray(1)
        # glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(12))

    def destroy(self):
        glDeleteVertexArrays(1, (self.vao,))
        glDeleteBuffers(1, (self.vbo,))

class ObjMesh:

    def __init__(self, filename):
        # x, y, z, s, t, nx, ny, nz, tangent, bitangent, model(instanced)
        self.vertices = self.loadMesh(filename)
        self.vertex_count = len(self.vertices)//14
        self.vertices = np.array(self.vertices, dtype=np.float32)

        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)
        offset = 0
        #position
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 56, ctypes.c_void_p(offset))
        offset += 12
        #texture
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 56, ctypes.c_void_p(offset))
        offset += 8
        #normal
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 56, ctypes.c_void_p(offset))
        offset += 12
        #tangent
        glEnableVertexAttribArray(3)
        glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, 56, ctypes.c_void_p(offset))
        offset += 12
        #bitangent
        glEnableVertexAttribArray(4)
        glVertexAttribPointer(4, 3, GL_FLOAT, GL_FALSE, 56, ctypes.c_void_p(offset))
        offset += 12
    
    def loadMesh(self, filename):

        #raw, unassembled data
        v = []
        vt = []
        vn = []
        
        #final, assembled and packed result
        vertices = []

        #open the obj file and read the data
        with open(filename,'r') as f:
            line = f.readline()
            while line:
                firstSpace = line.find(" ")
                flag = line[0:firstSpace]
                if flag=="v":
                    #vertex
                    line = line.replace("v ","")
                    line = line.split(" ")
                    l = [float(x) for x in line]
                    v.append(l)
                elif flag=="vt":
                    #texture coordinate
                    line = line.replace("vt ","")
                    line = line.split(" ")
                    l = [float(x) for x in line]
                    vt.append(l)
                elif flag=="vn":
                    #normal
                    line = line.replace("vn ","")
                    line = line.split(" ")
                    l = [float(x) for x in line]
                    vn.append(l)
                elif flag=="f":
                    #face, three or more vertices in v/vt/vn form
                    line = line.replace("f ","")
                    line = line.replace("\n","")
                    #get the individual vertices for each line
                    line = line.split(" ")
                    faceVertices = []
                    faceTextures = []
                    faceNormals = []
                    for vertex in line:
                        #break out into [v,vt,vn],
                        #correct for 0 based indexing.
                        l = vertex.split("/")
                        position = int(l[0]) - 1
                        faceVertices.append(v[position])
                        texture = int(l[1]) - 1
                        faceTextures.append(vt[texture])
                        normal = int(l[2]) - 1
                        faceNormals.append(vn[normal])
                    # obj file uses triangle fan format for each face individually.
                    # unpack each face
                    triangles_in_face = len(line) - 2

                    vertex_order = []
                    """
                        eg. 0,1,2,3 unpacks to vertices: [0,1,2,0,2,3]
                    """
                    for i in range(triangles_in_face):
                        vertex_order.append(0)
                        vertex_order.append(i+1)
                        vertex_order.append(i+2)
                    # calculate tangent and bitangent for point
                    # how do model positions relate to texture positions?
                    point1 = faceVertices[vertex_order[0]]
                    point2 = faceVertices[vertex_order[1]]
                    point3 = faceVertices[vertex_order[2]]
                    uv1 = faceTextures[vertex_order[0]]
                    uv2 = faceTextures[vertex_order[1]]
                    uv3 = faceTextures[vertex_order[2]]
                    #direction vectors
                    deltaPos1 = [point2[i] - point1[i] for i in range(3)]
                    deltaPos2 = [point3[i] - point1[i] for i in range(3)]
                    deltaUV1 = [uv2[i] - uv1[i] for i in range(2)]
                    deltaUV2 = [uv3[i] - uv1[i] for i in range(2)]
                    # calculate
                    den = 1
                    #den = 1 / (deltaUV1[0] * deltaUV2[1] - deltaUV2[0] * deltaUV1[1])
                    tangent = []
                    #tangent x
                    tangent.append(den * (deltaUV2[1] * deltaPos1[0] - deltaUV1[1] * deltaPos2[0]))
                    #tangent y
                    tangent.append(den * (deltaUV2[1] * deltaPos1[1] - deltaUV1[1] * deltaPos2[1]))
                    #tangent z
                    tangent.append(den * (deltaUV2[1] * deltaPos1[2] - deltaUV1[1] * deltaPos2[2]))
                    bitangent = []
                    #bitangent x
                    bitangent.append(den * (-deltaUV2[0] * deltaPos1[0] + deltaUV1[0] * deltaPos2[0]))
                    #bitangent y
                    bitangent.append(den * (-deltaUV2[0] * deltaPos1[1] + deltaUV1[0] * deltaPos2[1]))
                    #bitangent z
                    bitangent.append(den * (-deltaUV2[0] * deltaPos1[2] + deltaUV1[0] * deltaPos2[2]))
                    for i in vertex_order:
                        for x in faceVertices[i]:
                            vertices.append(x)
                        for x in faceTextures[i]:
                            vertices.append(x)
                        for x in faceNormals[i]:
                            vertices.append(x)
                        for x in tangent:
                            vertices.append(x)
                        for x in bitangent:
                            vertices.append(x)
                line = f.readline()
        return vertices
    
    def destroy(self):
        glDeleteVertexArrays(1, (self.vao,))
        glDeleteBuffers(1,(self.vbo,))

###############################################################################

window = initialize_glfw()
myApp = App(window)