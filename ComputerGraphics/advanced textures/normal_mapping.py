import glfw
import glfw.GLFW as GLFW_CONSTANTS
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram,compileShader
import numpy as np
import pyrr
import ctypes
from PIL import Image, ImageOps

############################## Constants ######################################


OBJECT_SOLAR  = 0
OBJECT_HULL =  1
OBJECT_DISH = 2
OBJECT_SKY = 3


SCREEN_WIDTH = 640
SCREEN_HEIGHT = 480

RETURN_ACTION_CONTINUE = 0
RETURN_ACTION_EXIT = 1

#0: debug, 1: production
GAME_MODE = 0
LIGHT_COUNT = 8

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

class Entity:


    def __init__(self, position, eulers, eulerVelocity):

        self.position = np.array(position, dtype=np.float32)
        self.eulers = np.array(eulers, dtype=np.float32)
        self.eulerVelocity = np.array(eulerVelocity, dtype=np.float32)


class Hull:


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


        self.entitys: dict[int,list[Entity]] = {}

        self.entitys[OBJECT_SOLAR] = []

        for i in range(10):
            position = [-1, ((-1 + i * 0.25)), -2]
            self.entitys[OBJECT_SOLAR].append(Entity(position, [0, 90, 0], [0, 0, 0]))
        
        i = 0
        for i in range(15):
            position = [-1 - 0.17, ((-1 + i * 0.25)), (-2 + (1 * 0.5))]
            self.entitys[OBJECT_SOLAR].append(Entity(position, [0, 90, 0], [0, 0, 0]))
            
        i = 0
        for i in range(10):
            position = [-1 - 0.34, ((-1 + i * 0.25)), (-2 + (2 * 0.5))]
            self.entitys[OBJECT_SOLAR].append(Entity(position, [0, 90, 0], [0, 0, 0]))

        k = 0
        for i in range(18):
            for j in range(3):
                if i < 10:
                    position = [-1 - (j * 0.17), ((4.5 + i * 0.25)), (-2 + (j * 0.5))]
                elif i > 9 and j == 1:
                    position = [-1 - (j * 0.17), ((4.5  - (k * 0.25))), (-2 + (j * 0.5))]
                    k+=1
                self.entitys[OBJECT_SOLAR].append(Entity(position, [0, 90, 0], [0, 0, 0]))
                
        self.entitys[OBJECT_HULL] = [
            Entity(
                position = [-1,-4,-0.5],
                eulers = [0,90,0],
                eulerVelocity = [0,0,0],
            )
        ]


        self.entitys[OBJECT_DISH] = [
            Entity(
                position = [-2.5,6,-0.5],
                eulers = [-20,0,-90],
                eulerVelocity = [0,0,0],
            )
        ]

        i = 0
        m = 0
        self.lights = []

        for i in range(LIGHT_COUNT):
            if(i < (LIGHT_COUNT/2)):
                position = [0,3+i,0.2+i]
            else:
                position = [2,6+i, 0.2 + m]
                m+=1
            self.lights.append(Light(position,color = [1, 1, 1]))
            # Light(
            #     color = [1, 1, 1]
            # )  

        self.player = Player(
            position = [0, -5, 0],
            eulers = [0, 0, 120]
        )
    
    def update(self, rate: float) -> None:
        return
        # for objectType,objectList in self.entitys.items():
        #     for entity in objectList:
        #         entity.eulers = np.mod(
        #             entity.eulers + entity.eulerVelocity * rate, 
        #             [360, 360, 360], 
        #             dtype=np.float32
        #         )

##################################### Control #################################

class App:


    def __init__(self, window):

        self.window = window

        self.lastTime = 0
        self.currentTime = 0
        self.numFrames = 0
        self.frameTime = 0

        self.scene = Scene()

        self.engine = Engine(self.scene,self.window)

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


    def __init__(self, scene, window):

        #initialise opengl
        glClearColor(0.1, 0.1, 0.1, 1)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_CULL_FACE)
        glCullFace(GL_BACK)

        self.create_framebuffers(window)
        self.create_assets(scene)
        self.set_up_shaders()

    def create_assets(self,scene):
        #create assets

        self.meshes: dict[int, ObjMesh] = {
            OBJECT_SOLAR: ObjMesh("models/solar.obj",0),
            OBJECT_HULL: ObjMesh("models/hull.obj",0),
            OBJECT_DISH:ObjMesh("models/dish.obj",0)
        }

        self.materials: dict[int, Material] = {
            OBJECT_SOLAR: Material("goldBrick", "png"),
            OBJECT_HULL: Material("hull", "png"),
            OBJECT_DISH: Material("dish", "png"),
        }

        self.entityTransforms = {}
        self.entityTransformVBO = {}

        for objectType,objectList in scene.entitys.items():
            mesh = self.meshes[objectType]
            print(objectType)
            #generate position buffer
            self.entityTransforms[objectType] = np.array([
                pyrr.matrix44.create_identity(dtype = np.float32)

                for i in range(len(objectList))
            ], dtype=np.float32)

            glBindVertexArray(mesh.vao)
            self.entityTransformVBO[objectType] = glGenBuffers(1)
            glBindBuffer(
                GL_ARRAY_BUFFER, 
                self.entityTransformVBO[objectType]
            )
            glBufferData(
                GL_ARRAY_BUFFER, 
                self.entityTransforms[objectType].nbytes, 
                self.entityTransforms[objectType], 
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
        
        #SCREEN SHADER 
        self.screen_shader = self.createShader("shaders/simple_post_vertex.txt", "shaders/screen_fragment.txt")
        glUseProgram(self.screen_shader)
        glUniform1i(glGetUniformLocation(self.screen_shader, "material"), 0)
        glUniform1i(glGetUniformLocation(self.screen_shader, "bright_material"), 1)

        #ADDING BLOOM
        self.bloom_blur_shader = self.createShader("shaders/simple_post_vertex.txt", "shaders/bloom_blur_fragment.txt")
        glUseProgram(self.bloom_blur_shader)
        glUniform1i(glGetUniformLocation(self.bloom_blur_shader, "material"), 0)
        glUniform1i(glGetUniformLocation(self.bloom_blur_shader, "bright_material"), 1)

        self.bloom_transfer_shader = self.createShader("shaders/simple_post_vertex.txt", "shaders/bloom_transfer_fragment.txt")
        glUseProgram(self.bloom_transfer_shader)
        glUniform1i(glGetUniformLocation(self.bloom_transfer_shader, "material"), 0)
        glUniform1i(glGetUniformLocation(self.bloom_transfer_shader, "bright_material"), 1)

        self.bloom_resolve_shader = self.createShader("shaders/simple_post_vertex.txt", "shaders/bloom_resolve_fragment.txt")
        glUseProgram(self.bloom_resolve_shader)
        glUniform1i(glGetUniformLocation(self.bloom_resolve_shader, "material"), 0)
        glUniform1i(glGetUniformLocation(self.bloom_resolve_shader, "bright_material"), 1)


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

    #ADDED BRIGHT MATERIAL TO FRAGMENT
        glUniform1i(
            glGetUniformLocation(
                self.shaderTextured, "bright_material"
                ), 4
            )


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


    


    def create_framebuffers(self,window):

        (self.w,self.h) = glfw.get_framebuffer_size(window)

        self.colorBuffers = [Colorbuffer(self.w, self.h), Colorbuffer(self.w, self.h)]
        self.depthBuffer = DepthStencilbuffer(self.w, self.h)
        self.framebuffers = Framebuffer(
            colorAttachments = [self.colorBuffers[0], self.colorBuffers[1]], 
            depthBuffer = self.depthBuffer
        )

    def createShader(self, vertexFilepath, fragmentFilepath):

        with open(vertexFilepath,'r') as f:
            vertex_src = f.readlines()

        with open(fragmentFilepath,'r') as f:
            fragment_src = f.readlines()

        shader = compileProgram(compileShader(vertex_src, GL_VERTEX_SHADER),
                            compileShader(fragment_src, GL_FRAGMENT_SHADER))
    
        
        return shader

    def draw(self, scene):

        # glBindFramebuffer(GL_FRAMEBUFFER, self.framebuffers.fbo)
        # glDrawBuffers(2, (GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1))
        # glClearColor(0,0,0.5,0)
        # glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        # glEnable(GL_DEPTH_TEST)
    
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
        

        for objectType,objectList in scene.entitys.items():

            for i, entity in enumerate(objectList):
                model_transform = pyrr.matrix44.create_identity(dtype=np.float32)
                model_transform = pyrr.matrix44.multiply(
                    m1=model_transform, 
                    m2=pyrr.matrix44.create_from_eulers(
                        eulers=np.radians(entity.eulers), dtype=np.float32
                    )
                )
                model_transform = pyrr.matrix44.multiply(
                    m1=model_transform, 
                    m2=pyrr.matrix44.create_from_translation(
                        vec=np.array(entity.position),dtype=np.float32
                    )
                )
                self.entityTransforms[objectType][i] = model_transform
            
            glBindVertexArray(self.meshes[objectType].vao)
            glBindBuffer(
                GL_ARRAY_BUFFER, 
                self.entityTransformVBO[objectType]
            )
            glBufferData(GL_ARRAY_BUFFER, self.entityTransforms[objectType].nbytes, self.entityTransforms[objectType], GL_STATIC_DRAW)
           
            self.materials[objectType].use()

            glDrawArraysInstanced(GL_TRIANGLES, 0, self.meshes[objectType].vertex_count, len(objectList))
            

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
        

            # glBindFramebuffer(GL_FRAMEBUFFER, 0)
            # glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            # glDisable(GL_DEPTH_TEST)

            # glUseProgram(self.bloom_blur_shader)
            # glActiveTexture(GL_TEXTURE0)
            # glBindTexture(GL_TEXTURE_2D, self.colorBuffers[0].texture)
            # glActiveTexture(GL_TEXTURE1)
            # glBindTexture(GL_TEXTURE_2D, self.colorBuffers[1].texture)
            # glBindVertexArray(self.screen.vao)

            # glUseProgram(self.bloom_transfer_shader)
            # glActiveTexture(GL_TEXTURE0)
            # glBindTexture(GL_TEXTURE_2D, self.colorBuffers[0].texture)
            # glActiveTexture(GL_TEXTURE1)
            # glBindTexture(GL_TEXTURE_2D, self.colorBuffers[1].texture)
            # glBindVertexArray(self.screen.vao)

        glFlush()

    def quit(self):
        self.entity_mesh.destroy()
        self.light_mesh.destroy()
        self.gold_texture.destroy()
        glDeleteBuffers(1, (self.entityTransformVBO,))
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



class TexturedQuad:

    def __init__(self, x, y, w, h):
        self.vertices = (
            x - w, y + h, 0, 1,
            x - w, y - h, 0, 0,
            x + w, y - h, 1, 0,

            x - w, y + h, 0, 1,
            x + w, y - h, 1, 0,
            x + w, y + h, 1, 1
        )
        self.vertices = np.array(self.vertices, dtype=np.float32)

        self.vertex_count = 6
        
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)

        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(0))

        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(8))
    
    def destroy(self):
        glDeleteVertexArrays(1, (self.vao,))
        glDeleteBuffers(1, (self.vbo,))




class DepthStencilbuffer:
    """
        A simple depth buffer which
        can be rendered to and read from.
    """

    
    def __init__(self, w: int, h: int):
        """
            Initialise the framebuffer.

            Parameters:
                w: the width of the screen
                h: the height of the screen
        """

        #create and bind, a render buffer is like a texture which can
        # be written to and read from, but not sampled (ie. not smooth)
        self.texture = glGenRenderbuffers(1)
        glBindRenderbuffer(GL_RENDERBUFFER, self.texture)
        #preallocate space, we'll use 24 bits for depth and 8 for stencil
        glRenderbufferStorage(
            GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, w, h
        )
        glBindRenderbuffer(GL_RENDERBUFFER,0)

class Colorbuffer:
    """
        A simple color buffer which
        can be rendered to and read from.
    """

    
    def __init__(self, w: int, h: int):
        """
            Initialise the colorbuffer.

            Parameters:
                w: the width of the screen
                h: the height of the screen
        """
        
        #create and bind the color buffer
        self.texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        #preallocate space
        glTexImage2D(
            GL_TEXTURE_2D, 0, GL_RGB, 
            w, h,
            0, GL_RGB, GL_UNSIGNED_BYTE, None
        )
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glBindTexture(GL_TEXTURE_2D, 0)
    

class Framebuffer:
    """
        A simple framebuffer object, holds a color buffer and depth buffer which
        can be rendered to and read from.
    """

    
    def __init__(self, 
                 colorAttachments: list[Colorbuffer], 
                 depthBuffer: DepthStencilbuffer):
        """
            Initialise the framebuffer.

            Parameters:
                w: the width of the screen
                h: the height of the screen
        """
        
        self.fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        
        for i,colorBuffer in enumerate(colorAttachments):
            glFramebufferTexture2D(GL_FRAMEBUFFER, 
                                GL_COLOR_ATTACHMENT0 + i, 
                                GL_TEXTURE_2D, colorBuffer.texture, 0)
        
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, 
                                    GL_RENDERBUFFER, depthBuffer.texture)

        glBindFramebuffer(GL_FRAMEBUFFER, 0)

class ObjMesh:

    def __init__(self, filename,split:int):
        # x, y, z, s, t, nx, ny, nz, tangent, bitangent, model(instanced)
        self.vertices = self.loadMesh(filename,split)
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
    
    def loadMesh(self, filename,split:int):

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
                        if(split == 1):
                            l = vertex.split("//")
                        else:
                            l = vertex.split("/")
                        
                        position = int(l[0]) - 1
                        faceVertices.append(v[position])
                        texture = int(l[1]) - 1
                        if(texture < len(vt)):
                            faceTextures.append(vt[texture])
                        if(len(l) > 2):
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


class MaterialCubemap(Material):

    def __init__(self, filepath):

        super().__init__(GL_TEXTURE_CUBE_MAP, 0)

        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        #load textures
        with Image.open(f"{filepath}_left.png", mode = "r") as img:
            image_width,image_height = img.size
            img = img.convert('RGBA')
            img_data = bytes(img.tobytes())
            glTexImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_Y,0,GL_RGBA8,image_width,image_height,0,GL_RGBA,GL_UNSIGNED_BYTE,img_data)
        
        with Image.open(f"{filepath}_right.png", mode = "r") as img:
            image_width,image_height = img.size
            img = ImageOps.flip(img)
            img = ImageOps.mirror(img)
            img = img.convert('RGBA')
            img_data = bytes(img.tobytes())
            glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_Y,0,GL_RGBA8,image_width,image_height,0,GL_RGBA,GL_UNSIGNED_BYTE,img_data)
        
        with Image.open(f"{filepath}_top.png", mode = "r") as img:
            image_width,image_height = img.size
            img = img.rotate(90)
            img = img.convert('RGBA')
            img_data = bytes(img.tobytes())
            glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_Z,0,GL_RGBA8,image_width,image_height,0,GL_RGBA,GL_UNSIGNED_BYTE,img_data)

        with Image.open(f"{filepath}_bottom.png", mode = "r") as img:
            image_width,image_height = img.size
            img = img.convert('RGBA')
            img_data = bytes(img.tobytes())
            glTexImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_Z,0,GL_RGBA8,image_width,image_height,0,GL_RGBA,GL_UNSIGNED_BYTE,img_data)
        
        with Image.open(f"{filepath}_back.png", mode = "r") as img:
            image_width,image_height = img.size
            img = img.rotate(-90)
            img = img.convert('RGBA')
            img_data = bytes(img.tobytes())
            glTexImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_X,0,GL_RGBA8,image_width,image_height,0,GL_RGBA,GL_UNSIGNED_BYTE,img_data)

        with Image.open(f"{filepath}_front.png", mode = "r") as img:
            image_width,image_height = img.size
            img = img.rotate(90)
            img = img.convert('RGBA')
            img_data = bytes(img.tobytes())
            glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X,0,GL_RGBA8,image_width,image_height,0,GL_RGBA,GL_UNSIGNED_BYTE,img_data)


###############################################################################

window = initialize_glfw()
myApp = App(window)