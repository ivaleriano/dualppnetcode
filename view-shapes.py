"""
For FSL run:
```
python view-shapes.py \
    /mnt/nas/Data_Neuro/ADNI3/FSL/FSL_out/1012942/1012942_all_fast_firstseg.nii.gz \
    -m /mnt/nas/Data_Neuro/ADNI3/FSL/FSL_out/1012942/1012942-L_Hipp_first.vtk \
    --volume-as-surface
```

For FreeSurfer run:
```
python view-shapes.py \
    /mnt/nas/Data_Neuro/ADNI2/standardized/1012942/mri/aseg_wimt.mgz \
    --volume-as-surface
´``
"""
import argparse
import tempfile
from pathlib import Path
from typing import Tuple

import nibabel as nib
import numpy as np
import vtk

COLOR_PALETTE = (
    # np.array([0.0, 0.0, 0.0]),
    # Dark2
    # np.array([117,112,179]) / 255,
    # np.array([27,158,119]) / 255,
    # np.array([217,95,2]) / 255,
    # Set1
    # np.array([55,126,184]) / 255,
    # np.array([55,126,184]) / 255,
    # np.array([55,126,184]) / 255,
    np.array([0.9, 0.9, 0.9]),
    # np.array([77,175,74]) / 255,
    np.array([55,126,184]) / 255,
    np.array([255,127,0]) / 255,
)


def MakeAnnotatedCubeActor(colors):
    """
    :param colors: Used to determine the cube color.
    :return: The annotated cube actor.
    """
    # A cube with labeled faces.
    cube = vtk.vtkAnnotatedCubeActor()
    cube.SetXPlusFaceText('R')  # Right
    cube.SetXMinusFaceText('L')  # Left
    cube.SetYPlusFaceText('A')  # Anterior
    cube.SetYMinusFaceText('P')  # Posterior
    cube.SetZPlusFaceText('S')  # Superior/Cranial
    cube.SetZMinusFaceText('I')  # Inferior/Caudal
    cube.SetFaceTextScale(0.5)
    cube.GetCubeProperty().SetColor(colors.GetColor3d('Gainsboro'))

    cube.GetTextEdgesProperty().SetColor(colors.GetColor3d('LightSlateGray'))

    # Change the vector text colors.
    cube.GetXPlusFaceProperty().SetColor(colors.GetColor3d('Tomato'))
    cube.GetXMinusFaceProperty().SetColor(colors.GetColor3d('Tomato'))
    cube.GetYPlusFaceProperty().SetColor(colors.GetColor3d('DeepSkyBlue'))
    cube.GetYMinusFaceProperty().SetColor(colors.GetColor3d('DeepSkyBlue'))
    cube.GetZPlusFaceProperty().SetColor(colors.GetColor3d('SeaGreen'))
    cube.GetZMinusFaceProperty().SetColor(colors.GetColor3d('SeaGreen'))
    return cube


def load_mgz(input_image_file_name):
    img = nib.load(str(input_image_file_name))
    nii_img = nib.Nifti1Image(img.get_fdata(), img.affine)

    A = nii_img.affine
    qmat = vtk.vtkMatrix4x4()
    for i in range(4):
        for j in range(4):
            v = A[i, j]
            qmat.SetElement(i, j, v)

    with tempfile.NamedTemporaryFile(suffix=".nii") as tempf:
        nib.save(nii_img, tempf.name)

        reader = vtk.vtkNIFTIImageReader()
        reader.SetFileName(tempf.name)
        reader.Update()

    return reader, qmat


def load_nifti(filename):
    reader = vtk.vtkNIFTIImageReader()
    reader.SetFileName(str(filename))
    reader.Update()
    header = reader.GetNIFTIHeader()

    mat = np.eye(4)
    mat[:3, 3] = (
        header.GetQOffsetX(),
        header.GetQOffsetY(),
        header.GetQOffsetZ(),
    )

    qmat = vtk.vtkMatrix4x4()
    for i in range(4):
        for j in range(4):
            qmat.SetElement(i, j, mat[i, j])

    return reader, qmat


###################################################
#
#  LOAD ORIGINAL SEG AND RUN MARCHING CUBES
#
###################################################

def load_segmentation_map(filename: Path, labelValue: int) -> vtk.vtkImageData:
    if filename.suffix == ".mgz":
        reader, qmat = load_mgz(filename)
    else:
        reader, qmat = load_nifti(filename)

    # Get structure with selected label
    threshold = vtk.vtkImageThreshold()
    threshold.SetInputConnection(reader.GetOutputPort())
    threshold.ThresholdBetween(labelValue, labelValue)
    threshold.ReplaceInOn()
    threshold.SetInValue(1)  # set all values below 400 to 0
    threshold.ReplaceOutOn()
    threshold.SetOutValue(0)  # set all values above 400 to 1
    threshold.Update()

    label_map = vtk.vtkImageData()
    label_map.DeepCopy(threshold.GetOutput())
    label_map2 = vtk.vtkImageData()
    label_map2.ShallowCopy(label_map)
    label_map2.SetOrigin(0, 0, 0)
    label_map2.SetSpacing(1.0, 1.0, 1.0)

    # print(qmat)
    # print(label_map.GetBounds())
    return label_map2


def mesh_from_segmentation_map(label_map2: vtk.vtkImageData, smoothing: bool = False) -> vtk.vtkPolyData:
    dmc = vtk.vtkDiscreteMarchingCubes()
    dmc.SetInputData(label_map2)
    dmc.ComputeGradientsOff()
    dmc.ComputeNormalsOff()
    dmc.GenerateValues(1, 1, 1)
    dmc.Update()

    if smoothing:
        # smoothFilter = vtk.vtkWindowedSincPolyDataFilter()
        smoothFilter = vtk.vtkSmoothPolyDataFilter()
        smoothFilter.SetInputConnection(dmc.GetOutputPort())
        smoothFilter.SetNumberOfIterations(10)
        smoothFilter.SetRelaxationFactor(0.05)
        smoothFilter.Update()
    else:
        smoothFilter = dmc

    triangulator = vtk.vtkTriangleFilter()
    triangulator.SetInputConnection(smoothFilter.GetOutputPort())
    triangulator.Update()

    # subdiv = vtk.vtkButterflySubdivisionFilter()
    # subdiv.SetInputConnection(triangulator.GetOutputPort())
    # subdiv.Update()

    # only keep points and their connections and
    # discard color and normals etc.
    polydata = triangulator.GetOutput()
    polygonPolyData = vtk.vtkPolyData()
    polygonPolyData.SetPoints(polydata.GetPoints())
    polygonPolyData.SetPolys(polydata.GetPolys())

    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(polygonPolyData)
    normals.SetFeatureAngle(30.0)
    normals.ComputePointNormalsOn()
    normals.SplittingOff()
    normals.Update()

    normalsPolyData = vtk.vtkPolyData()
    normalsPolyData.DeepCopy(normals.GetOutput())

    return normalsPolyData


def load_fsl_mesh(filename):
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(str(filename))
    reader.Update()

    normals = vtk.vtkPolyDataNormals()
    normals.SetInputConnection(reader.GetOutputPort())
    normals.SetFeatureAngle(30.0)
    normals.ComputePointNormalsOn()
    normals.SplittingOff()
    normals.Update()

    normalsPolyData = vtk.vtkPolyData()
    normalsPolyData.DeepCopy(normals.GetOutput())

    # https://git.fmrib.ox.ac.uk/fsl/fsleyes/fsleyes/-/blob/190a18aecf1edca9354e61f9cf8f39e028864fb9/fsleyes/tests/__init__.py#L670
    translation = vtk.vtkTransform()
    translation.SetMatrix(
        [
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, -1, 255,  # flip axis
            0, 0, 0, 1,
        ]
    )

    transformFilter = vtk.vtkTransformPolyDataFilter()
    transformFilter.SetInputData(normalsPolyData)
    transformFilter.SetTransform(translation)
    transformFilter.Update()

    return transformFilter.GetOutput()


def create_mesh_actor(
    poly_data: vtk.vtkPolyData,
    color: Tuple[float, float, float],
    respresentation: str,
):
    meshMapper = vtk.vtkPolyDataMapper()
    meshMapper.SetInputData(poly_data)

    meshActor = vtk.vtkActor()
    meshActor.SetMapper(meshMapper)
    meshActor.GetProperty().SetColor(color)
    meshActor.GetProperty().SetInterpolationToFlat()
    if respresentation == "wireframe":
        # meshActor.GetProperty().SetRepresentationToWireframe()

        # from https://kitware.github.io/vtk-examples/site/Python/Modelling/DelaunayMesh/
        extract = vtk.vtkExtractEdges()
        extract.SetInputData(poly_data)
        tubes = vtk.vtkTubeFilter()
        tubes.SetInputConnection(extract.GetOutputPort())
        tubes.SetRadius(0.05)
        tubes.SetNumberOfSides(6)
        mapEdges = vtk.vtkPolyDataMapper()
        mapEdges.SetInputConnection(tubes.GetOutputPort())
        edgeActor = vtk.vtkActor()
        edgeActor.SetMapper(mapEdges)
        edgeActor.GetProperty().SetColor(color)
        edgeActor.GetProperty().SetSpecularColor(1, 1, 1)
        edgeActor.GetProperty().SetSpecular(0.3)
        edgeActor.GetProperty().SetSpecularPower(20)
        edgeActor.GetProperty().SetAmbient(0.2)
        edgeActor.GetProperty().SetDiffuse(0.8)

        ball = vtk.vtkSphereSource()
        ball.SetRadius(0.15)
        ball.SetThetaResolution(12)
        ball.SetPhiResolution(12)
        balls = vtk.vtkGlyph3D()
        balls.SetInputData(poly_data)
        balls.SetSourceConnection(ball.GetOutputPort())
        mapBalls = vtk.vtkPolyDataMapper()
        mapBalls.SetInputConnection(balls.GetOutputPort())
        ballActor = vtk.vtkActor()
        ballActor.SetMapper(mapBalls)
        ballActor.GetProperty().SetColor(color)
        ballActor.GetProperty().SetSpecularColor(1, 1, 1)
        ballActor.GetProperty().SetSpecular(0.3)
        ballActor.GetProperty().SetSpecularPower(20)
        ballActor.GetProperty().SetAmbient(0.2)
        ballActor.GetProperty().SetDiffuse(0.8)

        meshActor = [edgeActor, ballActor]
    elif respresentation == "points":
        meshActor.GetProperty().SetRepresentationToPoints()
        meshActor.GetProperty().RenderPointsAsSpheresOn()
        meshActor.GetProperty().SetPointSize(5.0)
    elif respresentation == "surface":
        meshActor.GetProperty().SetRepresentationToSurface()
    # meshActor.GetProperty().SetOpacity(0.95)

    return meshActor


def create_volume_renderer_from_label_map(label_map: vtk.vtkImageData, color: Tuple[float, float, float]):
    # The following class is used to store transparencyv-values for later retrival.
    # In our case, we want the value 0 to be completly opaque
    alphaChannelFunc = vtk.vtkPiecewiseFunction()
    alphaChannelFunc.AddPoint(0, 0.0)
    alphaChannelFunc.AddPoint(1, 1.0)

    # This class stores color data and can create color tables from a few color points.
    colorFunc = vtk.vtkColorTransferFunction()
    colorFunc.AddRGBPoint(0, *color)

    # The property describes how the data will look.
    volumeProperty = vtk.vtkVolumeProperty()
    volumeProperty.SetColor(colorFunc)
    volumeProperty.SetScalarOpacity(alphaChannelFunc)
    volumeProperty.ShadeOn()
    volumeProperty.SetInterpolationTypeToLinear()

    # The mapper / ray cast function know how to render the data
    volumeMapper = vtk.vtkGPUVolumeRayCastMapper()
    # volumeMapper.SetBlendModeToAdditive()
    volumeMapper.SetBlendModeToComposite()
    volumeMapper.SetInputData(label_map)

    # The volume holds the mapper and the property and
    # can be used to position/orient the volume
    volume = vtk.vtkVolume()
    volume.SetMapper(volumeMapper)
    volume.SetProperty(volumeProperty)

    return volume


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=Path, help="Path to FreeSurfer (*.mgz) or FSL segementation map (*.nii.gz).")
    parser.add_argument("-l", "--label", type=int, default=17, help="Label of structure to extract.")
    parser.add_argument("-m", "--mesh", type=Path, help="Path to FSL mesh (*.vtk).")
    parser.add_argument(
        "--volume-as-surface", action="store_true", default=False,
        help="Whether to render volumetric mask as surface mesh."
    )

    args = parser.parse_args()

    label_map = load_segmentation_map(args.filename, args.label)
    vol_mesh = mesh_from_segmentation_map(label_map)
    if args.mesh is not None:
        seg_mesh = load_fsl_mesh(args.mesh)
    else:
        seg_mesh = mesh_from_segmentation_map(label_map, smoothing=True)

    colors = vtk.vtkNamedColors()

    # Create a renderer for each view
    rens = [vtk.vtkRenderer(), vtk.vtkRenderer(), vtk.vtkRenderer()]
    for r in rens:
        # r.SetBackground(0.8, 0.8, 0.8)
        r.SetBackground(1.0, 1.0, 1.0)
    rens[0].SetViewport(0, 0, 1.0 / 3.0, 1)
    rens[1].SetViewport(1.0 / 3.0, 0, 2.0 / 3.0, 1)
    rens[2].SetViewport(2.0 / 3.0, 0, 1, 1)

    # An outline provides context around the data.
    outlineData = vtk.vtkOutlineFilter()
    outlineData.SetInputData(label_map)

    mapOutline = vtk.vtkPolyDataMapper()
    mapOutline.SetInputConnection(outlineData.GetOutputPort())

    outline = vtk.vtkActor()
    outline.SetMapper(mapOutline)
    outline.GetProperty().SetColor(colors.GetColor3d("Black"))

    # shared camera
    aCamera = vtk.vtkCamera()

    # Create render window
    renWin = vtk.vtkRenderWindow()

    # Create interactor
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)

    # create volume renderer
    if args.volume_as_surface:
        meshActor = create_mesh_actor(vol_mesh, COLOR_PALETTE[0], "surface")
        rens[0].AddActor(meshActor)
    else:
        volume = create_volume_renderer_from_label_map(label_map, COLOR_PALETTE[0])
        rens[0].AddVolume(volume)

    rens[0].SetActiveCamera(aCamera)
    renWin.AddRenderer(rens[0])

    # Axes Indicator
    axes = MakeAnnotatedCubeActor(colors)
    widget = vtk.vtkOrientationMarkerWidget()
    widget.SetOrientationMarker(axes)
    widget.SetInteractor(iren)
    widget.SetViewport(0.0, 0.0, 0.2, 0.2)
    widget.SetEnabled(1)
    widget.InteractiveOn()

    for i, rep in enumerate(("wireframe", "points"), start=1):
        meshActor = create_mesh_actor(seg_mesh, COLOR_PALETTE[i], rep)
        if isinstance(meshActor, list):
            for m in meshActor:
                rens[i].AddActor(m)
        else:
            rens[i].AddActor(meshActor)
        rens[i].SetActiveCamera(aCamera)
        renWin.AddRenderer(rens[i])

    renWin.SetSize(1800, 600)

    rens[0].ResetCamera()
    aCamera = rens[0].GetActiveCamera()
    aCamera.SetViewUp(0, 0, 1)
    aCamera.SetPosition(0, -1, 0)
    aCamera.SetFocalPoint(0, 0, 0)
    aCamera.ComputeViewPlaneNormal()
    # aCamera.Azimuth(180.0)  # front=anterior, left, right, up=superior
    aCamera.Azimuth(225.0)
    aCamera.Elevation(25.0)

    renWin.SetWindowName("Shape Representations")
    renWin.Render()

    def CheckAbort(obj, event):
        if obj.GetEventPending() != 0:
            obj.SetAbortRender(1)

    renWin.AddObserver("AbortCheckEvent", CheckAbort)

    rens[0].ResetCamera()
    # Actors are added to the renderer. An initial camera view is created.
    # The Dolly() method moves the camera towards the FocalPoint,
    # thereby enlarging the image.
    aCamera.Dolly(7.0)

    rens[0].ResetCameraClippingRange()

    style = vtk.vtkInteractorStyleTrackballCamera()
    iren.SetInteractorStyle(style)

    iren.Initialize()
    renWin.Render()

    iren.Start()


if __name__ == "__main__":
    main()
