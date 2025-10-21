import bpy

C, D = bpy.context, bpy.data


env_node = C.scene.world.node_tree.nodes['Environment Texture']
env_node.image = D.images.load('//env.exr')
