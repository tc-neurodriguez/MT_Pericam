import bioformats
reader = bioformats.get_image_reader(r"C:\Users\Tyler\Box\ReiterLab_Members\Tyler\Studies\MT_Pericam\2025-03-24\20x test 1k zstack equidistant voxels resonant_0002.oir")
image = reader.read()
print(image.shape)