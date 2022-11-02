/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the 'License');
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an 'AS IS' BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
import { ResizeNearestNeighbor, util, } from '@tensorflow/tfjs-core';
import { cast } from './Cast';
let wasmResizeNearestNeighbor;
function setup(backend) {
    wasmResizeNearestNeighbor = backend.wasm.cwrap(ResizeNearestNeighbor, null /*void*/, [
        'number',
        'number',
        'number',
        'number',
        'number',
        'number',
        'number',
        'number',
        'number',
        'number', // outId
    ]);
}
function resizeNearestNeighbor(args) {
    const { backend, inputs, attrs } = args;
    const { images } = inputs;
    const { alignCorners, halfPixelCenters, size } = attrs;
    const [newHeight, newWidth] = size;
    const [batch, oldHeight, oldWidth, numChannels] = images.shape;
    const outShape = [batch, newHeight, newWidth, numChannels];
    const out = backend.makeOutput(outShape, 'float32');
    if (util.sizeFromShape(images.shape) === 0) {
        return out;
    }
    let xData = backend.dataIdMap.get(images.dataId);
    let castedData;
    if (xData.dtype !== 'float32') {
        castedData = cast({
            backend,
            inputs: { x: images },
            attrs: { dtype: 'float32' },
        });
        xData = backend.dataIdMap.get(castedData.dataId);
    }
    const xId = xData.id;
    const outId = backend.dataIdMap.get(out.dataId).id;
    wasmResizeNearestNeighbor(xId, batch, oldHeight, oldWidth, numChannels, newHeight, newWidth, alignCorners ? 1 : 0, halfPixelCenters ? 1 : 0, outId);
    if (castedData != null) {
        backend.disposeData(castedData.dataId);
    }
    return out;
}
export const resizeNearestNeighborConfig = {
    kernelName: ResizeNearestNeighbor,
    backendName: 'wasm',
    setupFunc: setup,
    kernelFunc: resizeNearestNeighbor,
};
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiUmVzaXplTmVhcmVzdE5laWdoYm9yLmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vLi4vdGZqcy1iYWNrZW5kLXdhc20vc3JjL2tlcm5lbHMvUmVzaXplTmVhcmVzdE5laWdoYm9yLnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBOzs7Ozs7Ozs7Ozs7Ozs7R0FlRztBQUVILE9BQU8sRUFHTCxxQkFBcUIsRUFJckIsSUFBSSxHQUNMLE1BQU0sdUJBQXVCLENBQUM7QUFJL0IsT0FBTyxFQUFFLElBQUksRUFBRSxNQUFNLFFBQVEsQ0FBQztBQUU5QixJQUFJLHlCQVdLLENBQUM7QUFFVixTQUFTLEtBQUssQ0FBQyxPQUFvQjtJQUNqQyx5QkFBeUIsR0FBRyxPQUFPLENBQUMsSUFBSSxDQUFDLEtBQUssQ0FDNUMscUJBQXFCLEVBQ3JCLElBQUksQ0FBQyxRQUFRLEVBQ2I7UUFDRSxRQUFRO1FBQ1IsUUFBUTtRQUNSLFFBQVE7UUFDUixRQUFRO1FBQ1IsUUFBUTtRQUNSLFFBQVE7UUFDUixRQUFRO1FBQ1IsUUFBUTtRQUNSLFFBQVE7UUFDUixRQUFRLEVBQUUsUUFBUTtLQUNuQixDQUNGLENBQUM7QUFDSixDQUFDO0FBRUQsU0FBUyxxQkFBcUIsQ0FBQyxJQUk5QjtJQUNDLE1BQU0sRUFBRSxPQUFPLEVBQUUsTUFBTSxFQUFFLEtBQUssRUFBRSxHQUFHLElBQUksQ0FBQztJQUN4QyxNQUFNLEVBQUUsTUFBTSxFQUFFLEdBQUcsTUFBTSxDQUFDO0lBQzFCLE1BQU0sRUFBRSxZQUFZLEVBQUUsZ0JBQWdCLEVBQUUsSUFBSSxFQUFFLEdBQUcsS0FBSyxDQUFDO0lBRXZELE1BQU0sQ0FBQyxTQUFTLEVBQUUsUUFBUSxDQUFDLEdBQUcsSUFBSSxDQUFDO0lBRW5DLE1BQU0sQ0FBQyxLQUFLLEVBQUUsU0FBUyxFQUFFLFFBQVEsRUFBRSxXQUFXLENBQUMsR0FBRyxNQUFNLENBQUMsS0FBSyxDQUFDO0lBQy9ELE1BQU0sUUFBUSxHQUFHLENBQUMsS0FBSyxFQUFFLFNBQVMsRUFBRSxRQUFRLEVBQUUsV0FBVyxDQUFDLENBQUM7SUFFM0QsTUFBTSxHQUFHLEdBQUcsT0FBTyxDQUFDLFVBQVUsQ0FBQyxRQUFRLEVBQUUsU0FBUyxDQUFDLENBQUM7SUFDcEQsSUFBSSxJQUFJLENBQUMsYUFBYSxDQUFDLE1BQU0sQ0FBQyxLQUFLLENBQUMsS0FBSyxDQUFDLEVBQUU7UUFDMUMsT0FBTyxHQUFHLENBQUM7S0FDWjtJQUVELElBQUksS0FBSyxHQUFHLE9BQU8sQ0FBQyxTQUFTLENBQUMsR0FBRyxDQUFDLE1BQU0sQ0FBQyxNQUFNLENBQUMsQ0FBQztJQUNqRCxJQUFJLFVBQVUsQ0FBQztJQUNmLElBQUksS0FBSyxDQUFDLEtBQUssS0FBSyxTQUFTLEVBQUU7UUFDN0IsVUFBVSxHQUFHLElBQUksQ0FBQztZQUNoQixPQUFPO1lBQ1AsTUFBTSxFQUFFLEVBQUUsQ0FBQyxFQUFFLE1BQU0sRUFBRTtZQUNyQixLQUFLLEVBQUUsRUFBRSxLQUFLLEVBQUUsU0FBUyxFQUFFO1NBQzVCLENBQUMsQ0FBQztRQUNILEtBQUssR0FBRyxPQUFPLENBQUMsU0FBUyxDQUFDLEdBQUcsQ0FBQyxVQUFVLENBQUMsTUFBTSxDQUFDLENBQUM7S0FDbEQ7SUFFRCxNQUFNLEdBQUcsR0FBRyxLQUFLLENBQUMsRUFBRSxDQUFDO0lBQ3JCLE1BQU0sS0FBSyxHQUFHLE9BQU8sQ0FBQyxTQUFTLENBQUMsR0FBRyxDQUFDLEdBQUcsQ0FBQyxNQUFNLENBQUMsQ0FBQyxFQUFFLENBQUM7SUFFbkQseUJBQXlCLENBQ3ZCLEdBQUcsRUFDSCxLQUFLLEVBQ0wsU0FBUyxFQUNULFFBQVEsRUFDUixXQUFXLEVBQ1gsU0FBUyxFQUNULFFBQVEsRUFDUixZQUFZLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxFQUNwQixnQkFBZ0IsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLEVBQ3hCLEtBQUssQ0FDTixDQUFDO0lBRUYsSUFBSSxVQUFVLElBQUksSUFBSSxFQUFFO1FBQ3RCLE9BQU8sQ0FBQyxXQUFXLENBQUMsVUFBVSxDQUFDLE1BQU0sQ0FBQyxDQUFDO0tBQ3hDO0lBRUQsT0FBTyxHQUFHLENBQUM7QUFDYixDQUFDO0FBRUQsTUFBTSxDQUFDLE1BQU0sMkJBQTJCLEdBQWlCO0lBQ3ZELFVBQVUsRUFBRSxxQkFBcUI7SUFDakMsV0FBVyxFQUFFLE1BQU07SUFDbkIsU0FBUyxFQUFFLEtBQUs7SUFDaEIsVUFBVSxFQUFFLHFCQUF5QztDQUN0RCxDQUFDIiwic291cmNlc0NvbnRlbnQiOlsiLyoqXG4gKiBAbGljZW5zZVxuICogQ29weXJpZ2h0IDIwMTkgR29vZ2xlIExMQy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cbiAqIExpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSAnTGljZW5zZScpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gJ0FTIElTJyBCQVNJUyxcbiAqIFdJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuICogU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxuICogbGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5cbmltcG9ydCB7XG4gIEtlcm5lbENvbmZpZyxcbiAgS2VybmVsRnVuYyxcbiAgUmVzaXplTmVhcmVzdE5laWdoYm9yLFxuICBSZXNpemVOZWFyZXN0TmVpZ2hib3JBdHRycyxcbiAgUmVzaXplTmVhcmVzdE5laWdoYm9ySW5wdXRzLFxuICBUZW5zb3JJbmZvLFxuICB1dGlsLFxufSBmcm9tICdAdGVuc29yZmxvdy90ZmpzLWNvcmUnO1xuXG5pbXBvcnQgeyBCYWNrZW5kV2FzbSB9IGZyb20gJy4uL2JhY2tlbmRfd2FzbSc7XG5cbmltcG9ydCB7IGNhc3QgfSBmcm9tICcuL0Nhc3QnO1xuXG5sZXQgd2FzbVJlc2l6ZU5lYXJlc3ROZWlnaGJvcjogKFxuICB4SWQ6IG51bWJlcixcbiAgYmF0Y2g6IG51bWJlcixcbiAgb2xkSGVpZ2h0OiBudW1iZXIsXG4gIG9sZFdpZHRoOiBudW1iZXIsXG4gIG51bUNoYW5uZWxzOiBudW1iZXIsXG4gIG5ld0hlaWdodDogbnVtYmVyLFxuICBuZXdXaWR0aDogbnVtYmVyLFxuICBhbGlnbkNvcm5lcnM6IG51bWJlcixcbiAgaGFsZlBpeGVsQ2VudGVyczogbnVtYmVyLFxuICBvdXRJZDogbnVtYmVyXG4pID0+IHZvaWQ7XG5cbmZ1bmN0aW9uIHNldHVwKGJhY2tlbmQ6IEJhY2tlbmRXYXNtKTogdm9pZCB7XG4gIHdhc21SZXNpemVOZWFyZXN0TmVpZ2hib3IgPSBiYWNrZW5kLndhc20uY3dyYXAoXG4gICAgUmVzaXplTmVhcmVzdE5laWdoYm9yLFxuICAgIG51bGwgLyp2b2lkKi8sXG4gICAgW1xuICAgICAgJ251bWJlcicsIC8vIHhJZFxuICAgICAgJ251bWJlcicsIC8vIGJhdGNoXG4gICAgICAnbnVtYmVyJywgLy8gb2xkSGVpZ2h0XG4gICAgICAnbnVtYmVyJywgLy8gb2xkV2lkdGhcbiAgICAgICdudW1iZXInLCAvLyBudW1DaGFubmVsc1xuICAgICAgJ251bWJlcicsIC8vIG5ld0hlaWdodFxuICAgICAgJ251bWJlcicsIC8vIG5ld1dpZHRoXG4gICAgICAnbnVtYmVyJywgLy8gYWxpZ25Db3JuZXJzXG4gICAgICAnbnVtYmVyJywgLy8gaGFsZlBpeGVsQ2VudGVyc1xuICAgICAgJ251bWJlcicsIC8vIG91dElkXG4gICAgXVxuICApO1xufVxuXG5mdW5jdGlvbiByZXNpemVOZWFyZXN0TmVpZ2hib3IoYXJnczoge1xuICBiYWNrZW5kOiBCYWNrZW5kV2FzbTtcbiAgaW5wdXRzOiBSZXNpemVOZWFyZXN0TmVpZ2hib3JJbnB1dHM7XG4gIGF0dHJzOiBSZXNpemVOZWFyZXN0TmVpZ2hib3JBdHRycztcbn0pOiBUZW5zb3JJbmZvIHtcbiAgY29uc3QgeyBiYWNrZW5kLCBpbnB1dHMsIGF0dHJzIH0gPSBhcmdzO1xuICBjb25zdCB7IGltYWdlcyB9ID0gaW5wdXRzO1xuICBjb25zdCB7IGFsaWduQ29ybmVycywgaGFsZlBpeGVsQ2VudGVycywgc2l6ZSB9ID0gYXR0cnM7XG5cbiAgY29uc3QgW25ld0hlaWdodCwgbmV3V2lkdGhdID0gc2l6ZTtcblxuICBjb25zdCBbYmF0Y2gsIG9sZEhlaWdodCwgb2xkV2lkdGgsIG51bUNoYW5uZWxzXSA9IGltYWdlcy5zaGFwZTtcbiAgY29uc3Qgb3V0U2hhcGUgPSBbYmF0Y2gsIG5ld0hlaWdodCwgbmV3V2lkdGgsIG51bUNoYW5uZWxzXTtcblxuICBjb25zdCBvdXQgPSBiYWNrZW5kLm1ha2VPdXRwdXQob3V0U2hhcGUsICdmbG9hdDMyJyk7XG4gIGlmICh1dGlsLnNpemVGcm9tU2hhcGUoaW1hZ2VzLnNoYXBlKSA9PT0gMCkge1xuICAgIHJldHVybiBvdXQ7XG4gIH1cblxuICBsZXQgeERhdGEgPSBiYWNrZW5kLmRhdGFJZE1hcC5nZXQoaW1hZ2VzLmRhdGFJZCk7XG4gIGxldCBjYXN0ZWREYXRhO1xuICBpZiAoeERhdGEuZHR5cGUgIT09ICdmbG9hdDMyJykge1xuICAgIGNhc3RlZERhdGEgPSBjYXN0KHtcbiAgICAgIGJhY2tlbmQsXG4gICAgICBpbnB1dHM6IHsgeDogaW1hZ2VzIH0sXG4gICAgICBhdHRyczogeyBkdHlwZTogJ2Zsb2F0MzInIH0sXG4gICAgfSk7XG4gICAgeERhdGEgPSBiYWNrZW5kLmRhdGFJZE1hcC5nZXQoY2FzdGVkRGF0YS5kYXRhSWQpO1xuICB9XG5cbiAgY29uc3QgeElkID0geERhdGEuaWQ7XG4gIGNvbnN0IG91dElkID0gYmFja2VuZC5kYXRhSWRNYXAuZ2V0KG91dC5kYXRhSWQpLmlkO1xuXG4gIHdhc21SZXNpemVOZWFyZXN0TmVpZ2hib3IoXG4gICAgeElkLFxuICAgIGJhdGNoLFxuICAgIG9sZEhlaWdodCxcbiAgICBvbGRXaWR0aCxcbiAgICBudW1DaGFubmVscyxcbiAgICBuZXdIZWlnaHQsXG4gICAgbmV3V2lkdGgsXG4gICAgYWxpZ25Db3JuZXJzID8gMSA6IDAsXG4gICAgaGFsZlBpeGVsQ2VudGVycyA/IDEgOiAwLFxuICAgIG91dElkXG4gICk7XG5cbiAgaWYgKGNhc3RlZERhdGEgIT0gbnVsbCkge1xuICAgIGJhY2tlbmQuZGlzcG9zZURhdGEoY2FzdGVkRGF0YS5kYXRhSWQpO1xuICB9XG5cbiAgcmV0dXJuIG91dDtcbn1cblxuZXhwb3J0IGNvbnN0IHJlc2l6ZU5lYXJlc3ROZWlnaGJvckNvbmZpZzogS2VybmVsQ29uZmlnID0ge1xuICBrZXJuZWxOYW1lOiBSZXNpemVOZWFyZXN0TmVpZ2hib3IsXG4gIGJhY2tlbmROYW1lOiAnd2FzbScsXG4gIHNldHVwRnVuYzogc2V0dXAsXG4gIGtlcm5lbEZ1bmM6IHJlc2l6ZU5lYXJlc3ROZWlnaGJvciBhcyB7fSBhcyBLZXJuZWxGdW5jLFxufTtcbiJdfQ==