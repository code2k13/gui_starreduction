/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
import { Transform, util } from '@tensorflow/tfjs-core';
let wasmTransform;
function setup(backend) {
    wasmTransform = backend.wasm.cwrap(Transform, null /*void*/, [
        'number',
        'number',
        'bool',
        'number',
        'number',
        'number',
        'number',
        'number',
        'number',
        'array',
        'number',
        'array',
        'number',
        'number',
        'number',
        'number',
        'number' // outId
    ]);
}
function transform(args) {
    const { backend, inputs, attrs } = args;
    const { image, transforms } = inputs;
    const { interpolation, fillMode, fillValue, outputShape } = attrs;
    const [batch, imageHeight, imageWidth, numChannels] = image.shape;
    const [outHeight, outWidth] = outputShape != null ? outputShape : [imageHeight, imageWidth];
    const outShape = [batch, outHeight, outWidth,
        numChannels];
    const inputStrides = new Uint8Array(new Int32Array(util.computeStrides(image.shape)).buffer);
    const outputStrides = new Uint8Array(new Int32Array(util.computeStrides(outShape)).buffer);
    const out = backend.makeOutput(outShape, image.dtype);
    const outId = backend.dataIdMap.get(out.dataId).id;
    const imageData = backend.dataIdMap.get(image.dataId);
    const imageId = imageData.id;
    const transformsData = backend.dataIdMap.get(transforms.dataId);
    const transformsId = transformsData.id;
    const interpolationModeId = interpolation === 'nearest' ? 1 : 2;
    let fillModeId;
    switch (fillMode) {
        case 'constant':
            fillModeId = 1;
            break;
        case 'reflect':
            fillModeId = 2;
            break;
        case 'wrap':
            fillModeId = 3;
            break;
        case 'nearest':
            fillModeId = 4;
            break;
        default:
            fillModeId = 1;
            break;
    }
    wasmTransform(imageId, transformsId, (transforms.shape[0] > 1), batch, outHeight, outWidth, numChannels, imageWidth, imageHeight, inputStrides, image.shape.length - 1, outputStrides, outShape.length - 1, interpolationModeId, fillModeId, fillValue, outId);
    return out;
}
export const transformConfig = {
    kernelName: Transform,
    backendName: 'wasm',
    setupFunc: setup,
    kernelFunc: transform
};
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiVHJhbnNmb3JtLmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vLi4vdGZqcy1iYWNrZW5kLXdhc20vc3JjL2tlcm5lbHMvVHJhbnNmb3JtLnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBOzs7Ozs7Ozs7Ozs7Ozs7R0FlRztBQUVILE9BQU8sRUFBdUMsU0FBUyxFQUFtQyxJQUFJLEVBQUMsTUFBTSx1QkFBdUIsQ0FBQztBQUk3SCxJQUFJLGFBTTZELENBQUM7QUFFbEUsU0FBUyxLQUFLLENBQUMsT0FBb0I7SUFDakMsYUFBYSxHQUFHLE9BQU8sQ0FBQyxJQUFJLENBQUMsS0FBSyxDQUFDLFNBQVMsRUFBRSxJQUFJLENBQUMsUUFBUSxFQUFFO1FBQzNELFFBQVE7UUFDUixRQUFRO1FBQ1IsTUFBTTtRQUNOLFFBQVE7UUFDUixRQUFRO1FBQ1IsUUFBUTtRQUNSLFFBQVE7UUFDUixRQUFRO1FBQ1IsUUFBUTtRQUNSLE9BQU87UUFDUCxRQUFRO1FBQ1IsT0FBTztRQUNQLFFBQVE7UUFDUixRQUFRO1FBQ1IsUUFBUTtRQUNSLFFBQVE7UUFDUixRQUFRLENBQUcsUUFBUTtLQUNwQixDQUFDLENBQUM7QUFDTCxDQUFDO0FBRUQsU0FBUyxTQUFTLENBQ2QsSUFDMEU7SUFFNUUsTUFBTSxFQUFDLE9BQU8sRUFBRSxNQUFNLEVBQUUsS0FBSyxFQUFDLEdBQUcsSUFBSSxDQUFDO0lBQ3RDLE1BQU0sRUFBQyxLQUFLLEVBQUUsVUFBVSxFQUFDLEdBQUcsTUFBTSxDQUFDO0lBQ25DLE1BQU0sRUFBQyxhQUFhLEVBQUUsUUFBUSxFQUFFLFNBQVMsRUFBRSxXQUFXLEVBQUMsR0FBRyxLQUFLLENBQUM7SUFFaEUsTUFBTSxDQUFDLEtBQUssRUFBRSxXQUFXLEVBQUUsVUFBVSxFQUFFLFdBQVcsQ0FBQyxHQUFHLEtBQUssQ0FBQyxLQUFLLENBQUM7SUFDbEUsTUFBTSxDQUFDLFNBQVMsRUFBRSxRQUFRLENBQUMsR0FDdkIsV0FBVyxJQUFJLElBQUksQ0FBQyxDQUFDLENBQUMsV0FBVyxDQUFDLENBQUMsQ0FBQyxDQUFDLFdBQVcsRUFBRSxVQUFVLENBQUMsQ0FBQztJQUNsRSxNQUFNLFFBQVEsR0FDVixDQUFDLEtBQUssRUFBRSxTQUFTLEVBQUUsUUFBUTtRQUMxQixXQUFXLENBQXFDLENBQUM7SUFDdEQsTUFBTSxZQUFZLEdBQ2QsSUFBSSxVQUFVLENBQUMsSUFBSSxVQUFVLENBQUMsSUFBSSxDQUFDLGNBQWMsQ0FBQyxLQUFLLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxNQUFNLENBQUMsQ0FBQztJQUU1RSxNQUFNLGFBQWEsR0FDZixJQUFJLFVBQVUsQ0FBQyxJQUFJLFVBQVUsQ0FBQyxJQUFJLENBQUMsY0FBYyxDQUFDLFFBQVEsQ0FBQyxDQUFDLENBQUMsTUFBTSxDQUFDLENBQUM7SUFFekUsTUFBTSxHQUFHLEdBQUcsT0FBTyxDQUFDLFVBQVUsQ0FBQyxRQUFRLEVBQUUsS0FBSyxDQUFDLEtBQUssQ0FBQyxDQUFDO0lBQ3RELE1BQU0sS0FBSyxHQUFHLE9BQU8sQ0FBQyxTQUFTLENBQUMsR0FBRyxDQUFDLEdBQUcsQ0FBQyxNQUFNLENBQUMsQ0FBQyxFQUFFLENBQUM7SUFFbkQsTUFBTSxTQUFTLEdBQUcsT0FBTyxDQUFDLFNBQVMsQ0FBQyxHQUFHLENBQUMsS0FBSyxDQUFDLE1BQU0sQ0FBQyxDQUFDO0lBQ3RELE1BQU0sT0FBTyxHQUFHLFNBQVMsQ0FBQyxFQUFFLENBQUM7SUFFN0IsTUFBTSxjQUFjLEdBQUcsT0FBTyxDQUFDLFNBQVMsQ0FBQyxHQUFHLENBQUMsVUFBVSxDQUFDLE1BQU0sQ0FBQyxDQUFDO0lBQ2hFLE1BQU0sWUFBWSxHQUFHLGNBQWMsQ0FBQyxFQUFFLENBQUM7SUFFdkMsTUFBTSxtQkFBbUIsR0FBRyxhQUFhLEtBQUssU0FBUyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQztJQUNoRSxJQUFJLFVBQVUsQ0FBQztJQUNmLFFBQVEsUUFBUSxFQUFFO1FBQ2hCLEtBQUssVUFBVTtZQUNiLFVBQVUsR0FBRyxDQUFDLENBQUM7WUFDZixNQUFNO1FBQ1IsS0FBSyxTQUFTO1lBQ1osVUFBVSxHQUFHLENBQUMsQ0FBQztZQUNmLE1BQU07UUFDUixLQUFLLE1BQU07WUFDVCxVQUFVLEdBQUcsQ0FBQyxDQUFDO1lBQ2YsTUFBTTtRQUNSLEtBQUssU0FBUztZQUNaLFVBQVUsR0FBRyxDQUFDLENBQUM7WUFDZixNQUFNO1FBQ1I7WUFDRSxVQUFVLEdBQUcsQ0FBQyxDQUFDO1lBQ2YsTUFBTTtLQUNUO0lBRUQsYUFBYSxDQUNULE9BQU8sRUFBRSxZQUFZLEVBQUUsQ0FBQyxVQUFVLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxHQUFHLENBQUMsQ0FBQyxFQUFFLEtBQUssRUFBRSxTQUFTLEVBQ2xFLFFBQVEsRUFBRSxXQUFXLEVBQUUsVUFBVSxFQUFFLFdBQVcsRUFBRSxZQUFZLEVBQzVELEtBQUssQ0FBQyxLQUFLLENBQUMsTUFBTSxHQUFHLENBQUMsRUFBRSxhQUFhLEVBQUUsUUFBUSxDQUFDLE1BQU0sR0FBRyxDQUFDLEVBQzFELG1CQUFtQixFQUFFLFVBQVUsRUFBRSxTQUFTLEVBQUUsS0FBSyxDQUFDLENBQUM7SUFFdkQsT0FBTyxHQUFHLENBQUM7QUFDYixDQUFDO0FBRUQsTUFBTSxDQUFDLE1BQU0sZUFBZSxHQUFpQjtJQUMzQyxVQUFVLEVBQUUsU0FBUztJQUNyQixXQUFXLEVBQUUsTUFBTTtJQUNuQixTQUFTLEVBQUUsS0FBSztJQUNoQixVQUFVLEVBQUUsU0FBNkI7Q0FDMUMsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDIxIEdvb2dsZSBMTEMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cblxuaW1wb3J0IHtLZXJuZWxDb25maWcsIEtlcm5lbEZ1bmMsIFRlbnNvckluZm8sIFRyYW5zZm9ybSwgVHJhbnNmb3JtQXR0cnMsIFRyYW5zZm9ybUlucHV0cywgdXRpbH0gZnJvbSAnQHRlbnNvcmZsb3cvdGZqcy1jb3JlJztcblxuaW1wb3J0IHtCYWNrZW5kV2FzbX0gZnJvbSAnLi4vYmFja2VuZF93YXNtJztcblxubGV0IHdhc21UcmFuc2Zvcm06IChcbiAgICBpbWFnZUlkOiBudW1iZXIsIHRyYW5zZm9ybXNJZDogbnVtYmVyLCBpc0JhdGNoVHJhbnNmb3JtOiBib29sZWFuLFxuICAgIGJhdGNoOiBudW1iZXIsIG91dEhlaWdodDogbnVtYmVyLCBvdXRXaWR0aDogbnVtYmVyLCBudW1DaGFubmVsczogbnVtYmVyLFxuICAgIGltYWdlV2lkdGg6IG51bWJlciwgaW1hZ2VIZWlnaHQ6IG51bWJlciwgaW5wdXRTdHJpZGVzOiBVaW50OEFycmF5LFxuICAgIGlucHV0U3RyaWRlc0xlbmd0aDogbnVtYmVyLCBvdXRwdXRTdHJpZGVzOiBVaW50OEFycmF5LFxuICAgIG91dHB1dFN0cmlkZXNMZW5ndGg6IG51bWJlciwgaW50ZXJwb2xhdGlvbk1vZGVJZDogbnVtYmVyLFxuICAgIGZpbGxNb2RlSWQ6IG51bWJlciwgZmlsbFZhbHVlOiBudW1iZXIsIG91dElkOiBudW1iZXIpID0+IHZvaWQ7XG5cbmZ1bmN0aW9uIHNldHVwKGJhY2tlbmQ6IEJhY2tlbmRXYXNtKTogdm9pZCB7XG4gIHdhc21UcmFuc2Zvcm0gPSBiYWNrZW5kLndhc20uY3dyYXAoVHJhbnNmb3JtLCBudWxsIC8qdm9pZCovLCBbXG4gICAgJ251bWJlcicsICAvLyBpbWFnZUlkXG4gICAgJ251bWJlcicsICAvLyB0cmFuc2Zvcm1zSWRcbiAgICAnYm9vbCcsICAgIC8vIGlzQmF0Y2hUcmFuc2Zvcm1cbiAgICAnbnVtYmVyJywgIC8vIGJhdGNoXG4gICAgJ251bWJlcicsICAvLyBvdXRIZWlnaHRcbiAgICAnbnVtYmVyJywgIC8vIG91dFdpZHRoXG4gICAgJ251bWJlcicsICAvLyBudW1DaGFubmVsc1xuICAgICdudW1iZXInLCAgLy8gaW1hZ2VXaWR0aFxuICAgICdudW1iZXInLCAgLy8gaW1hZ2VIZWlnaHRcbiAgICAnYXJyYXknLCAgIC8vIGlucHV0U3RyaWRlc1xuICAgICdudW1iZXInLCAgLy8gaW5wdXRTdHJpZGVzTGVuZ3RoXG4gICAgJ2FycmF5JywgICAvLyBvdXRwdXRTdHJpZGVzXG4gICAgJ251bWJlcicsICAvLyBvdXRwdXRTdHJpZGVzTGVuZ3RoXG4gICAgJ251bWJlcicsICAvLyBpbnRlcnBvbGF0aW9uTW9kZUlkXG4gICAgJ251bWJlcicsICAvLyBmaWxsTW9kZUlkXG4gICAgJ251bWJlcicsICAvLyBmaWxsVmFsdWVcbiAgICAnbnVtYmVyJyAgIC8vIG91dElkXG4gIF0pO1xufVxuXG5mdW5jdGlvbiB0cmFuc2Zvcm0oXG4gICAgYXJnczpcbiAgICAgICAge2JhY2tlbmQ6IEJhY2tlbmRXYXNtLCBpbnB1dHM6IFRyYW5zZm9ybUlucHV0cywgYXR0cnM6IFRyYW5zZm9ybUF0dHJzfSk6XG4gICAgVGVuc29ySW5mbyB7XG4gIGNvbnN0IHtiYWNrZW5kLCBpbnB1dHMsIGF0dHJzfSA9IGFyZ3M7XG4gIGNvbnN0IHtpbWFnZSwgdHJhbnNmb3Jtc30gPSBpbnB1dHM7XG4gIGNvbnN0IHtpbnRlcnBvbGF0aW9uLCBmaWxsTW9kZSwgZmlsbFZhbHVlLCBvdXRwdXRTaGFwZX0gPSBhdHRycztcblxuICBjb25zdCBbYmF0Y2gsIGltYWdlSGVpZ2h0LCBpbWFnZVdpZHRoLCBudW1DaGFubmVsc10gPSBpbWFnZS5zaGFwZTtcbiAgY29uc3QgW291dEhlaWdodCwgb3V0V2lkdGhdID1cbiAgICAgIG91dHB1dFNoYXBlICE9IG51bGwgPyBvdXRwdXRTaGFwZSA6IFtpbWFnZUhlaWdodCwgaW1hZ2VXaWR0aF07XG4gIGNvbnN0IG91dFNoYXBlID1cbiAgICAgIFtiYXRjaCwgb3V0SGVpZ2h0LCBvdXRXaWR0aCxcbiAgICAgICBudW1DaGFubmVsc10gYXMgW251bWJlciwgbnVtYmVyLCBudW1iZXIsIG51bWJlcl07XG4gIGNvbnN0IGlucHV0U3RyaWRlcyA9XG4gICAgICBuZXcgVWludDhBcnJheShuZXcgSW50MzJBcnJheSh1dGlsLmNvbXB1dGVTdHJpZGVzKGltYWdlLnNoYXBlKSkuYnVmZmVyKTtcblxuICBjb25zdCBvdXRwdXRTdHJpZGVzID1cbiAgICAgIG5ldyBVaW50OEFycmF5KG5ldyBJbnQzMkFycmF5KHV0aWwuY29tcHV0ZVN0cmlkZXMob3V0U2hhcGUpKS5idWZmZXIpO1xuXG4gIGNvbnN0IG91dCA9IGJhY2tlbmQubWFrZU91dHB1dChvdXRTaGFwZSwgaW1hZ2UuZHR5cGUpO1xuICBjb25zdCBvdXRJZCA9IGJhY2tlbmQuZGF0YUlkTWFwLmdldChvdXQuZGF0YUlkKS5pZDtcblxuICBjb25zdCBpbWFnZURhdGEgPSBiYWNrZW5kLmRhdGFJZE1hcC5nZXQoaW1hZ2UuZGF0YUlkKTtcbiAgY29uc3QgaW1hZ2VJZCA9IGltYWdlRGF0YS5pZDtcblxuICBjb25zdCB0cmFuc2Zvcm1zRGF0YSA9IGJhY2tlbmQuZGF0YUlkTWFwLmdldCh0cmFuc2Zvcm1zLmRhdGFJZCk7XG4gIGNvbnN0IHRyYW5zZm9ybXNJZCA9IHRyYW5zZm9ybXNEYXRhLmlkO1xuXG4gIGNvbnN0IGludGVycG9sYXRpb25Nb2RlSWQgPSBpbnRlcnBvbGF0aW9uID09PSAnbmVhcmVzdCcgPyAxIDogMjtcbiAgbGV0IGZpbGxNb2RlSWQ7XG4gIHN3aXRjaCAoZmlsbE1vZGUpIHtcbiAgICBjYXNlICdjb25zdGFudCc6XG4gICAgICBmaWxsTW9kZUlkID0gMTtcbiAgICAgIGJyZWFrO1xuICAgIGNhc2UgJ3JlZmxlY3QnOlxuICAgICAgZmlsbE1vZGVJZCA9IDI7XG4gICAgICBicmVhaztcbiAgICBjYXNlICd3cmFwJzpcbiAgICAgIGZpbGxNb2RlSWQgPSAzO1xuICAgICAgYnJlYWs7XG4gICAgY2FzZSAnbmVhcmVzdCc6XG4gICAgICBmaWxsTW9kZUlkID0gNDtcbiAgICAgIGJyZWFrO1xuICAgIGRlZmF1bHQ6XG4gICAgICBmaWxsTW9kZUlkID0gMTtcbiAgICAgIGJyZWFrO1xuICB9XG5cbiAgd2FzbVRyYW5zZm9ybShcbiAgICAgIGltYWdlSWQsIHRyYW5zZm9ybXNJZCwgKHRyYW5zZm9ybXMuc2hhcGVbMF0gPiAxKSwgYmF0Y2gsIG91dEhlaWdodCxcbiAgICAgIG91dFdpZHRoLCBudW1DaGFubmVscywgaW1hZ2VXaWR0aCwgaW1hZ2VIZWlnaHQsIGlucHV0U3RyaWRlcyxcbiAgICAgIGltYWdlLnNoYXBlLmxlbmd0aCAtIDEsIG91dHB1dFN0cmlkZXMsIG91dFNoYXBlLmxlbmd0aCAtIDEsXG4gICAgICBpbnRlcnBvbGF0aW9uTW9kZUlkLCBmaWxsTW9kZUlkLCBmaWxsVmFsdWUsIG91dElkKTtcblxuICByZXR1cm4gb3V0O1xufVxuXG5leHBvcnQgY29uc3QgdHJhbnNmb3JtQ29uZmlnOiBLZXJuZWxDb25maWcgPSB7XG4gIGtlcm5lbE5hbWU6IFRyYW5zZm9ybSxcbiAgYmFja2VuZE5hbWU6ICd3YXNtJyxcbiAgc2V0dXBGdW5jOiBzZXR1cCxcbiAga2VybmVsRnVuYzogdHJhbnNmb3JtIGFzIHt9IGFzIEtlcm5lbEZ1bmNcbn07XG4iXX0=