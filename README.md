# RebuilderAI - 송찬영 과제

## Task

"첨부드린 다시점 이미지에 대해 아래 두 모델을 돌려서 textured mesh를 각각 취득해 보는 과업입니다.
각 이미지는 에셋을 중앙에 배치하고 정확한 +x, -x, +y, -y, +z, -z 축 방향 perspective camera로 렌더링되었습니다."

- https://github.com/estheryang11/ReconViaGen
- https://github.com/NVlabs/nvdiffrec

## Input data

<h3>Multi-View Images</h3>
<table width="100%">
  <tr>
    <th width="16%"><div align="center">Front</div></th>
    <th width="16%"><div align="center">Back</div></th>
    <th width="16%"><div align="center">Left</div></th>
    <th width="16%"><div align="center">Right</div></th>
    <th width="16%"><div align="center">Top</div></th>
    <th width="16%"><div align="center">Bottom</div></th>
  </tr>
  <tr>
    <td align="center"><img src="./input/multi_views/front.png" width="100%" alt="Front"></td>
    <td align="center"><img src="./input/multi_views/back.png" width="100%" alt="Back"></td>
    <td align="center"><img src="./input/multi_views/left.png" width="100%" alt="Left"></td>
    <td align="center"><img src="./input/multi_views/right.png" width="100%" alt="Right"></td>
    <td align="center"><img src="./input/multi_views/top.png" width="100%" alt="Top"></td>
    <td align="center"><img src="./input/multi_views/bottom.png" width="100%" alt="Bottom"></td>
  </tr>
</table>

<h3>Masks</h3>
<table width="100%">
  <tr>
    <th width="16%"><div align="center">Front</div></th>
    <th width="16%"><div align="center">Back</div></th>
    <th width="16%"><div align="center">Left</div></th>
    <th width="16%"><div align="center">Right</div></th>
    <th width="16%"><div align="center">Top</div></th>
    <th width="16%"><div align="center">Bottom</div></th>
  </tr>
  <tr>
    <td align="center"><img src="./input/masks/front.png" width="100%" alt="Front Mask"></td>
    <td align="center"><img src="./input/masks/back.png" width="100%" alt="Back Mask"></td>
    <td align="center"><img src="./input/masks/left.png" width="100%" alt="Left Mask"></td>
    <td align="center"><img src="./input/masks/right.png" width="100%" alt="Right Mask"></td>
    <td align="center"><img src="./input/masks/top.png" width="100%" alt="Top Mask"></td>
    <td align="center"><img src="./input/masks/bottom.png" width="100%" alt="Bottom Mask"></td>
  </tr>
</table>

## Output Results

-   [ReconViaGen](./ReconViaGen/notes.md)
-   [nvdiffrec](./nvdiffrec/notes.md)
