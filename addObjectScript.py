import bpy
import os



#export_dir="home/Desktop/tacto/examples"




class TestPanel(bpy.types.Panel):
    bl_label = "Test Panel"
    bl_idname = "PT_TestPanel"
    bl_space_type = 'VIEW_3D' #VIEW, IMAGE_EDITOR 
    bl_region_type = 'UI'
    bl_category = 'My 1st Add-on' #Tool
    
    def draw(self, context):
        layout = self.layout
        
        row = layout.row()
        row.label(text="Sample Text", icon= 'CUBE')
        
        row = layout.row()
        row.operator("mesh.primitive_cube_add")
        
        row = layout.row()
        row.operator("mesh.primitive_cube_add")
        
        
def register():
    bpy.utils.register_class(TestPanel)
    
    
def unregister():
    bpy.utils.register_class(TestPanel)
    
def make_surface(point_type, point_size, spacing_x, spacing_y,adjust_x,adjust_y):
    #add a cube
#    bpy.ops.mesh.primitive_cube_add(size=2, location=(0,0,0))

    #add the "surface"
    bpy.ops.mesh.primitive_cube_add(size=2, location=(0,0,0))
    surface = bpy.context.active_object
    surface.name = "Surface"
    surface.scale.z = 0.05
    
    if point_size == '1':
        point_radius = 0.17
        point_cylinder_radius = 0.08
        doughnut_radius=0.23
        monkey_size = 0.45
    elif point_size == '2':
        point_radius = 0.19
        point_cylinder_radius = 0.12
        doughnut_radius=0.17
        monkey_size = 0.5375
    elif point_size == '3':
        point_radius = 0.21
        point_cylinder_radius = 0.16
        doughnut_radius=0.19
        monkey_size = 0.625
    elif point_size == '4':
        point_radius = 0.23
        point_cylinder_radius = 0.20
        doughnut_radius=0.21
        monkey_size = 0.7125
    elif point_size == '5':
        point_radius = 0.25
        point_cylinder_radius = 0.24
        doughnut_radius=0.23
        monkey_size = 0.8
        
    spacing = 0.6
    start_x = -spacing
    start_y = -spacing
#    z_height = surface.location.z + surface.scale.z + point_radius
    z_height = surface.location.z + surface.scale.z 
    
    
    #create 9 points (3*3)
    
    for i in range(3):
        for j in range(3):
            x = start_x + i * (spacing+spacing_x) + adjust_x
            y = start_y + j * (spacing+spacing_y) + adjust_y
            
            #spacing
            #spacing(spacing_case,i,j)
            
            
            if(point_type == 'Sphere'):    
                bpy.ops.mesh.primitive_uv_sphere_add(
                    radius=point_radius,
                    location=(x, y, z_height)
                )
            elif(point_type == 'Icosphere'):
                bpy.ops.mesh.primitive_ico_sphere_add(
                location=(x, y, z_height), 
                radius = point_radius
                )
            elif(point_type == 'Cube'):
                cube_height = point_radius-surface.scale.z
                bpy.ops.mesh.primitive_cube_add(
                    size=point_radius,
                    location=(x, y, cube_height)
                )
            elif(point_type == 'Cylinder'):
                bpy.ops.mesh.primitive_cylinder_add(
                    radius = point_cylinder_radius,
                    depth = 0.5,
                    location = (x, y, z_height)
                    )
            elif(point_type == 'Triangle'):
                bpy.ops.mesh.primitive_cylinder_add(
                    radius = point_cylinder_radius,
                    depth = 0.5,
                    location = (x, y, z_height),
                    vertices = 3
                    )
                bpy.context.object.name = 'Triangle'
            elif(point_type == 'Vertices_5'):
                bpy.ops.mesh.primitive_cylinder_add(
                    radius = point_cylinder_radius,
                    depth = 0.5,
                    location = (x, y, z_height),
                    vertices = 5
                    )
                bpy.context.object.name = 'Vertices_5'
            elif(point_type == 'Vertices_8'):
                bpy.ops.mesh.primitive_cylinder_add(
                    radius = point_cylinder_radius,
                    depth = 0.5,
                    location = (x, y, z_height),
                    vertices = 8
                    )
                bpy.context.object.name = 'Vertices_8'
            elif(point_type == 'Doughnut'):
                bpy.ops.mesh.primitive_torus_add(align='WORLD', location=(x, y, z_height), rotation=(0, 0, 0), major_radius=doughnut_radius, minor_radius=0.2)
                bpy.context.object.name = 'Doughnut'
            elif(point_type == 'Monkey'):
                bpy.ops.mesh.primitive_monkey_add(size= monkey_size, enter_editmode=False, align='WORLD', location=(x, y, z_height), scale=(0.1, 0.1, 0.1), rotation=(-1.57,0,0))
                bpy.context.object.name = 'Monkey'
            elif(point_type == 'Monkey_rotated'):
                bpy.ops.mesh.primitive_monkey_add(size= monkey_size, enter_editmode=False, align='WORLD', location=(x, y, z_height), scale=(0.1, 0.1, 0.1), rotation=(-1.57,-1.57,0))
                bpy.context.object.name = 'Monkey_rotated'

            
def remove_extra(target, point_type):
    
    extra_sphere_num = target
    if extra_sphere_num == 0 :
        extra_sphere = bpy.data.objects.get(f'{point_type}')
        
    elif extra_sphere_num <10:
    #target is always lower than 9, then no worries for more than digit objects and extra logic.
        extra_sphere = bpy.data.objects.get(f'{point_type}.00{extra_sphere_num}')
    if extra_sphere:
        bpy.data.objects.remove(extra_sphere, do_unlink=True)
        
Letters = {'a':[2],
'b':[1,2],
'c':[2,5],
'd':[2,4,5],
'e':[2,4],
'f':[1,2,5],
'g':[1,2,4,5],
'h':[1,2,4],
'i':[1,5],
'j':[1,4,5],
'k':[0,2],
'l':[0,1,2],
'm':[0,2,5],
'n':[0,2,4,5],
'o':[0,2,4],
'p':[0,1,2,5],
'q':[0,1,2,4,5],
'r':[0,1,2,4],
's':[0,1,5],
't':[0,1,4,5],
'u':[0,2,3],
'v':[0,1,2,3],
'w':[1,3,4,5],
'x':[0,2,3,5],
'y':[0,2,3,4,5],
'z':[0,2,3,4],
}

#Input here the letters wanted to be produced.
alphabet = 'r'
point_types = {'Sphere','Icosphere','Cube','Cylinder','Triangle','Vertices_5','Vertices_8','Doughnut','Monkey', 'Monkey_rotated'}
point_sizes = {'1','2','3','4','5'}
#more spacing adjust number can be added to generate more samples if it was needed later. (for example {0, 0.01, 0.02, 0.03, 0.04, -0.01, -0.02, -0.03, -0.04})
spacing_x = {0, 0.04, -0.04}
spacing_y = {0, 0.04, -0.04}
adjust_x = {0, 0.1, -0.1}
adjust_y = {0, 0.1, -0.1}
    
if __name__ == "__main__":
    
    #for point_type,point_size, point_spacing * in range(1,pointsizes)/ spacing
    
    for letter in alphabet:    
        for point_type in point_types:
            for point_size in point_sizes:
                for sp_x in spacing_x:
                    for sp_y in spacing_y:
                        for adj_x in adjust_x:
                            for adj_y in adjust_y: 
                                
                                #clean the scene
                                bpy.ops.object.select_all(action='SELECT')
                                bpy.ops.object.delete(use_global=False)
                                
                        #        register()
                                
                                #object mode
                                if bpy.ops.object.mode_set.poll():
                                        bpy.ops.object.mode_set(mode='OBJECT')
                                
                                make_surface(point_type, point_size, sp_x, sp_y, adj_x, adj_y)
                                #later change it to point so be able to change it with cubes, triangles and all.
                                
                            
                                for i in range(9):
                                    if not(i in Letters[letter]):
                                        remove_extra(i, point_type)

                                #Output folder
                                home_dir = os.path.expanduser(f'~/Desktop/tacto/examples/objects/samples/letter-{letter}')
                                fn = f'letter-{letter}.obj'
                                target_file = os.path.join(home_dir, fn)
                                if not os.path.exists(home_dir):
                                    os.makedirs(home_dir, exist_ok = True)
                                
                                base, extension = os.path.splitext(target_file)
                                counter = 2
                                while os.path.exists(target_file):
                                    target_file = f'{base}{counter}{extension}'
                                    counter += 1
                                
                                
                            #    for obj in bpy.context.scene.objects:
                            #        if obj.type == 'MESH':
                            #            obj.select_set(True)
                            #            bpy.context.view_layer.objects.active = obj

                                bpy.ops.object.select_all(action='SELECT')
                                
                                
                                
                                bpy.ops.wm.obj_export(
                                    filepath = target_file,
                                    export_selected_objects= True,
                                    export_eval_mode='DAG_EVAL_VIEWPORT',
                                    export_triangulated_mesh = True
                                    )
                                
                
            