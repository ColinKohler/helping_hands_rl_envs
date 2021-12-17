import os


for obj in range(0, 89):
# for obj in range(0, 1):
    objname = str(obj).zfill(3)
    os.system('~/bullet3/bin/test_vhacd_gmake_x64_release --input {}/textured.obj --output {}/convex.obj'
              .format(objname, objname))

    print(objname)
