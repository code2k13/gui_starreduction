/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
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
import { backend_util, Conv2D } from '@tensorflow/tfjs-core';
let wasmConv2d;
function setup(backend) {
    wasmConv2d = backend.wasm.cwrap(Conv2D, null /* void */, [
        'number',
        'number',
        'number',
        'number',
        'number',
        'number',
        'number',
        'number',
        'number',
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
function conv2d(args) {
    const { inputs, attrs, backend } = args;
    const { x, filter } = inputs;
    const xId = backend.dataIdMap.get(x.dataId).id;
    const filterId = backend.dataIdMap.get(filter.dataId).id;
    const { strides, dilations, pad, dimRoundingMode, dataFormat } = attrs;
    const $dataFormat = backend_util.convertConv2DDataFormat(dataFormat);
    const convInfo = backend_util.computeConv2DInfo(x.shape, filter.shape, strides, dilations, pad, dimRoundingMode, false, $dataFormat);
    const filterHeight = convInfo.filterHeight;
    const filterWidth = convInfo.filterWidth;
    const padTop = convInfo.padInfo.top;
    const padRight = convInfo.padInfo.right;
    const padBottom = convInfo.padInfo.bottom;
    const padLeft = convInfo.padInfo.left;
    const dilationHeight = convInfo.dilationHeight;
    const dilationWidth = convInfo.dilationWidth;
    const strideHeight = convInfo.strideHeight;
    const strideWidth = convInfo.strideWidth;
    const inputChannels = convInfo.inChannels;
    const outputChannels = convInfo.outChannels;
    const isSamePad = convInfo.padInfo.type === 'SAME' ? 1 : 0;
    if (convInfo.dataFormat !== 'channelsLast') {
        throw new Error(`wasm backend Conv2D does not support dataFormat:'` +
            `${convInfo.dataFormat}'. Please use 'channelsLast'.`);
    }
    const out = backend.makeOutput(convInfo.outShape, 'float32');
    const outId = backend.dataIdMap.get(out.dataId).id;
    wasmConv2d(xId, x.shape[0], x.shape[1], x.shape[2], filterId, filterHeight, filterWidth, padTop, padRight, padBottom, padLeft, isSamePad, dilationHeight, dilationWidth, strideHeight, strideWidth, inputChannels, outputChannels, outId);
    return out;
}
export const conv2DConfig = {
    kernelName: Conv2D,
    backendName: 'wasm',
    setupFunc: setup,
    kernelFunc: conv2d
};
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiQ29udjJELmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vLi4vdGZqcy1iYWNrZW5kLXdhc20vc3JjL2tlcm5lbHMvQ29udjJELnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBOzs7Ozs7Ozs7Ozs7Ozs7R0FlRztBQUVILE9BQU8sRUFBQyxZQUFZLEVBQUUsTUFBTSxFQUFnRSxNQUFNLHVCQUF1QixDQUFDO0FBSTFILElBQUksVUFNc0IsQ0FBQztBQUUzQixTQUFTLEtBQUssQ0FBQyxPQUFvQjtJQUNqQyxVQUFVLEdBQUcsT0FBTyxDQUFDLElBQUksQ0FBQyxLQUFLLENBQUMsTUFBTSxFQUFFLElBQUksQ0FBQyxVQUFVLEVBQUU7UUFDdkQsUUFBUTtRQUNSLFFBQVE7UUFDUixRQUFRO1FBQ1IsUUFBUTtRQUNSLFFBQVE7UUFDUixRQUFRO1FBQ1IsUUFBUTtRQUNSLFFBQVE7UUFDUixRQUFRO1FBQ1IsUUFBUTtRQUNSLFFBQVE7UUFDUixRQUFRO1FBQ1IsUUFBUTtRQUNSLFFBQVE7UUFDUixRQUFRO1FBQ1IsUUFBUTtRQUNSLFFBQVE7UUFDUixRQUFRO1FBQ1IsUUFBUSxFQUFHLFFBQVE7S0FDcEIsQ0FBQyxDQUFDO0FBQ0wsQ0FBQztBQUVELFNBQVMsTUFBTSxDQUNYLElBQXNFO0lBQ3hFLE1BQU0sRUFBQyxNQUFNLEVBQUUsS0FBSyxFQUFFLE9BQU8sRUFBQyxHQUFHLElBQUksQ0FBQztJQUV0QyxNQUFNLEVBQUMsQ0FBQyxFQUFFLE1BQU0sRUFBQyxHQUFHLE1BQU0sQ0FBQztJQUMzQixNQUFNLEdBQUcsR0FBRyxPQUFPLENBQUMsU0FBUyxDQUFDLEdBQUcsQ0FBQyxDQUFDLENBQUMsTUFBTSxDQUFDLENBQUMsRUFBRSxDQUFDO0lBQy9DLE1BQU0sUUFBUSxHQUFHLE9BQU8sQ0FBQyxTQUFTLENBQUMsR0FBRyxDQUFDLE1BQU0sQ0FBQyxNQUFNLENBQUMsQ0FBQyxFQUFFLENBQUM7SUFFekQsTUFBTSxFQUFDLE9BQU8sRUFBRSxTQUFTLEVBQUUsR0FBRyxFQUFFLGVBQWUsRUFBRSxVQUFVLEVBQUMsR0FBRyxLQUFLLENBQUM7SUFDckUsTUFBTSxXQUFXLEdBQUcsWUFBWSxDQUFDLHVCQUF1QixDQUFDLFVBQVUsQ0FBQyxDQUFDO0lBQ3JFLE1BQU0sUUFBUSxHQUFHLFlBQVksQ0FBQyxpQkFBaUIsQ0FDMUMsQ0FBYyxDQUFDLEtBQUssRUFBRyxNQUFtQixDQUFDLEtBQUssRUFBRSxPQUFPLEVBQUUsU0FBUyxFQUNyRSxHQUFHLEVBQUUsZUFBZSxFQUFFLEtBQUssRUFBRSxXQUFXLENBQUMsQ0FBQztJQUU5QyxNQUFNLFlBQVksR0FBRyxRQUFRLENBQUMsWUFBWSxDQUFDO0lBQzNDLE1BQU0sV0FBVyxHQUFHLFFBQVEsQ0FBQyxXQUFXLENBQUM7SUFDekMsTUFBTSxNQUFNLEdBQUcsUUFBUSxDQUFDLE9BQU8sQ0FBQyxHQUFHLENBQUM7SUFDcEMsTUFBTSxRQUFRLEdBQUcsUUFBUSxDQUFDLE9BQU8sQ0FBQyxLQUFLLENBQUM7SUFDeEMsTUFBTSxTQUFTLEdBQUcsUUFBUSxDQUFDLE9BQU8sQ0FBQyxNQUFNLENBQUM7SUFDMUMsTUFBTSxPQUFPLEdBQUcsUUFBUSxDQUFDLE9BQU8sQ0FBQyxJQUFJLENBQUM7SUFDdEMsTUFBTSxjQUFjLEdBQUcsUUFBUSxDQUFDLGNBQWMsQ0FBQztJQUMvQyxNQUFNLGFBQWEsR0FBRyxRQUFRLENBQUMsYUFBYSxDQUFDO0lBQzdDLE1BQU0sWUFBWSxHQUFHLFFBQVEsQ0FBQyxZQUFZLENBQUM7SUFDM0MsTUFBTSxXQUFXLEdBQUcsUUFBUSxDQUFDLFdBQVcsQ0FBQztJQUN6QyxNQUFNLGFBQWEsR0FBRyxRQUFRLENBQUMsVUFBVSxDQUFDO0lBQzFDLE1BQU0sY0FBYyxHQUFHLFFBQVEsQ0FBQyxXQUFXLENBQUM7SUFDNUMsTUFBTSxTQUFTLEdBQUcsUUFBUSxDQUFDLE9BQU8sQ0FBQyxJQUFJLEtBQUssTUFBTSxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQztJQUUzRCxJQUFJLFFBQVEsQ0FBQyxVQUFVLEtBQUssY0FBYyxFQUFFO1FBQzFDLE1BQU0sSUFBSSxLQUFLLENBQ1gsbURBQW1EO1lBQ25ELEdBQUcsUUFBUSxDQUFDLFVBQVUsK0JBQStCLENBQUMsQ0FBQztLQUM1RDtJQUVELE1BQU0sR0FBRyxHQUFHLE9BQU8sQ0FBQyxVQUFVLENBQUMsUUFBUSxDQUFDLFFBQVEsRUFBRSxTQUFTLENBQUMsQ0FBQztJQUM3RCxNQUFNLEtBQUssR0FBRyxPQUFPLENBQUMsU0FBUyxDQUFDLEdBQUcsQ0FBQyxHQUFHLENBQUMsTUFBTSxDQUFDLENBQUMsRUFBRSxDQUFDO0lBQ25ELFVBQVUsQ0FDTixHQUFHLEVBQUUsQ0FBQyxDQUFDLEtBQUssQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLEVBQUUsUUFBUSxFQUFFLFlBQVksRUFDL0QsV0FBVyxFQUFFLE1BQU0sRUFBRSxRQUFRLEVBQUUsU0FBUyxFQUFFLE9BQU8sRUFBRSxTQUFTLEVBQzVELGNBQWMsRUFBRSxhQUFhLEVBQUUsWUFBWSxFQUFFLFdBQVcsRUFBRSxhQUFhLEVBQ3ZFLGNBQWMsRUFBRSxLQUFLLENBQUMsQ0FBQztJQUMzQixPQUFPLEdBQUcsQ0FBQztBQUNiLENBQUM7QUFFRCxNQUFNLENBQUMsTUFBTSxZQUFZLEdBQWlCO0lBQ3hDLFVBQVUsRUFBRSxNQUFNO0lBQ2xCLFdBQVcsRUFBRSxNQUFNO0lBQ25CLFNBQVMsRUFBRSxLQUFLO0lBQ2hCLFVBQVUsRUFBRSxNQUEwQjtDQUN2QyxDQUFDIiwic291cmNlc0NvbnRlbnQiOlsiLyoqXG4gKiBAbGljZW5zZVxuICogQ29weXJpZ2h0IDIwMTkgR29vZ2xlIExMQy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cbiAqIExpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG4gKiB5b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG4gKiBZb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcbiAqXG4gKiBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcbiAqXG4gKiBVbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG4gKiBkaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG4gKiBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cbiAqIFNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbiAqIGxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuICogPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbiAqL1xuXG5pbXBvcnQge2JhY2tlbmRfdXRpbCwgQ29udjJELCBDb252MkRBdHRycywgQ29udjJESW5wdXRzLCBLZXJuZWxDb25maWcsIEtlcm5lbEZ1bmMsIFRlbnNvcjREfSBmcm9tICdAdGVuc29yZmxvdy90ZmpzLWNvcmUnO1xuXG5pbXBvcnQge0JhY2tlbmRXYXNtfSBmcm9tICcuLi9iYWNrZW5kX3dhc20nO1xuXG5sZXQgd2FzbUNvbnYyZDogKFxuICAgIHhJZDogbnVtYmVyLCBiYXRjaFNpemU6IG51bWJlciwgaW5wdXRIZWlnaHQ6IG51bWJlciwgaW5wdXRXaWR0aDogbnVtYmVyLFxuICAgIGZpbHRlcklkOiBudW1iZXIsIGZpbHRlckhlaWdodDogbnVtYmVyLCBmaWx0ZXJXaWR0aDogbnVtYmVyLCBwYWRUb3A6IG51bWJlcixcbiAgICBwYWRSaWdodDogbnVtYmVyLCBwYWRCb3R0b206IG51bWJlciwgcGFkTGVmdDogbnVtYmVyLCBpc1NhbWVQYWQ6IG51bWJlcixcbiAgICBkaWxhdGlvbkhlaWdodDogbnVtYmVyLCBkaWxhdGlvbldpZHRoOiBudW1iZXIsIHN0cmlkZUhlaWdodDogbnVtYmVyLFxuICAgIHN0cmlkZVdpZHRoOiBudW1iZXIsIGlucHV0Q2hhbm5lbHM6IG51bWJlciwgb3V0cHV0Q2hhbm5lbHM6IG51bWJlcixcbiAgICBvdXRJZDogbnVtYmVyKSA9PiB2b2lkO1xuXG5mdW5jdGlvbiBzZXR1cChiYWNrZW5kOiBCYWNrZW5kV2FzbSkge1xuICB3YXNtQ29udjJkID0gYmFja2VuZC53YXNtLmN3cmFwKENvbnYyRCwgbnVsbCAvKiB2b2lkICovLCBbXG4gICAgJ251bWJlcicsICAvLyB4SWRcbiAgICAnbnVtYmVyJywgIC8vIGJhdGNoU2l6ZVxuICAgICdudW1iZXInLCAgLy8gaW5wdXRIZWlnaHRcbiAgICAnbnVtYmVyJywgIC8vIGlucHV0V2lkdGhcbiAgICAnbnVtYmVyJywgIC8vIGZpbHRlcklkXG4gICAgJ251bWJlcicsICAvLyBmaWx0ZXJIZWlnaHRcbiAgICAnbnVtYmVyJywgIC8vIGZpbHRlcldpZHRoXG4gICAgJ251bWJlcicsICAvLyBwYWRUb3BcbiAgICAnbnVtYmVyJywgIC8vIHBhZFJpZ2h0XG4gICAgJ251bWJlcicsICAvLyBwYWRCb3R0b21cbiAgICAnbnVtYmVyJywgIC8vIHBhZExlZnRcbiAgICAnbnVtYmVyJywgIC8vIGlzU2FtZVBhZFxuICAgICdudW1iZXInLCAgLy8gZGlsYXRpb25IZWlnaHRcbiAgICAnbnVtYmVyJywgIC8vIGRpbGF0aW9uV2lkdGhcbiAgICAnbnVtYmVyJywgIC8vIHN0cmlkZUhlaWdodFxuICAgICdudW1iZXInLCAgLy8gc3RyaWRlV2lkdGhcbiAgICAnbnVtYmVyJywgIC8vIGlucHV0Q2hhbm5lbHNcbiAgICAnbnVtYmVyJywgIC8vIG91dHB1dENoYW5uZWxzXG4gICAgJ251bWJlcicsICAvLyBvdXRJZFxuICBdKTtcbn1cblxuZnVuY3Rpb24gY29udjJkKFxuICAgIGFyZ3M6IHtpbnB1dHM6IENvbnYyRElucHV0cywgYmFja2VuZDogQmFja2VuZFdhc20sIGF0dHJzOiBDb252MkRBdHRyc30pIHtcbiAgY29uc3Qge2lucHV0cywgYXR0cnMsIGJhY2tlbmR9ID0gYXJncztcblxuICBjb25zdCB7eCwgZmlsdGVyfSA9IGlucHV0cztcbiAgY29uc3QgeElkID0gYmFja2VuZC5kYXRhSWRNYXAuZ2V0KHguZGF0YUlkKS5pZDtcbiAgY29uc3QgZmlsdGVySWQgPSBiYWNrZW5kLmRhdGFJZE1hcC5nZXQoZmlsdGVyLmRhdGFJZCkuaWQ7XG5cbiAgY29uc3Qge3N0cmlkZXMsIGRpbGF0aW9ucywgcGFkLCBkaW1Sb3VuZGluZ01vZGUsIGRhdGFGb3JtYXR9ID0gYXR0cnM7XG4gIGNvbnN0ICRkYXRhRm9ybWF0ID0gYmFja2VuZF91dGlsLmNvbnZlcnRDb252MkREYXRhRm9ybWF0KGRhdGFGb3JtYXQpO1xuICBjb25zdCBjb252SW5mbyA9IGJhY2tlbmRfdXRpbC5jb21wdXRlQ29udjJESW5mbyhcbiAgICAgICh4IGFzIFRlbnNvcjREKS5zaGFwZSwgKGZpbHRlciBhcyBUZW5zb3I0RCkuc2hhcGUsIHN0cmlkZXMsIGRpbGF0aW9ucyxcbiAgICAgIHBhZCwgZGltUm91bmRpbmdNb2RlLCBmYWxzZSwgJGRhdGFGb3JtYXQpO1xuXG4gIGNvbnN0IGZpbHRlckhlaWdodCA9IGNvbnZJbmZvLmZpbHRlckhlaWdodDtcbiAgY29uc3QgZmlsdGVyV2lkdGggPSBjb252SW5mby5maWx0ZXJXaWR0aDtcbiAgY29uc3QgcGFkVG9wID0gY29udkluZm8ucGFkSW5mby50b3A7XG4gIGNvbnN0IHBhZFJpZ2h0ID0gY29udkluZm8ucGFkSW5mby5yaWdodDtcbiAgY29uc3QgcGFkQm90dG9tID0gY29udkluZm8ucGFkSW5mby5ib3R0b207XG4gIGNvbnN0IHBhZExlZnQgPSBjb252SW5mby5wYWRJbmZvLmxlZnQ7XG4gIGNvbnN0IGRpbGF0aW9uSGVpZ2h0ID0gY29udkluZm8uZGlsYXRpb25IZWlnaHQ7XG4gIGNvbnN0IGRpbGF0aW9uV2lkdGggPSBjb252SW5mby5kaWxhdGlvbldpZHRoO1xuICBjb25zdCBzdHJpZGVIZWlnaHQgPSBjb252SW5mby5zdHJpZGVIZWlnaHQ7XG4gIGNvbnN0IHN0cmlkZVdpZHRoID0gY29udkluZm8uc3RyaWRlV2lkdGg7XG4gIGNvbnN0IGlucHV0Q2hhbm5lbHMgPSBjb252SW5mby5pbkNoYW5uZWxzO1xuICBjb25zdCBvdXRwdXRDaGFubmVscyA9IGNvbnZJbmZvLm91dENoYW5uZWxzO1xuICBjb25zdCBpc1NhbWVQYWQgPSBjb252SW5mby5wYWRJbmZvLnR5cGUgPT09ICdTQU1FJyA/IDEgOiAwO1xuXG4gIGlmIChjb252SW5mby5kYXRhRm9ybWF0ICE9PSAnY2hhbm5lbHNMYXN0Jykge1xuICAgIHRocm93IG5ldyBFcnJvcihcbiAgICAgICAgYHdhc20gYmFja2VuZCBDb252MkQgZG9lcyBub3Qgc3VwcG9ydCBkYXRhRm9ybWF0OidgICtcbiAgICAgICAgYCR7Y29udkluZm8uZGF0YUZvcm1hdH0nLiBQbGVhc2UgdXNlICdjaGFubmVsc0xhc3QnLmApO1xuICB9XG5cbiAgY29uc3Qgb3V0ID0gYmFja2VuZC5tYWtlT3V0cHV0KGNvbnZJbmZvLm91dFNoYXBlLCAnZmxvYXQzMicpO1xuICBjb25zdCBvdXRJZCA9IGJhY2tlbmQuZGF0YUlkTWFwLmdldChvdXQuZGF0YUlkKS5pZDtcbiAgd2FzbUNvbnYyZChcbiAgICAgIHhJZCwgeC5zaGFwZVswXSwgeC5zaGFwZVsxXSwgeC5zaGFwZVsyXSwgZmlsdGVySWQsIGZpbHRlckhlaWdodCxcbiAgICAgIGZpbHRlcldpZHRoLCBwYWRUb3AsIHBhZFJpZ2h0LCBwYWRCb3R0b20sIHBhZExlZnQsIGlzU2FtZVBhZCxcbiAgICAgIGRpbGF0aW9uSGVpZ2h0LCBkaWxhdGlvbldpZHRoLCBzdHJpZGVIZWlnaHQsIHN0cmlkZVdpZHRoLCBpbnB1dENoYW5uZWxzLFxuICAgICAgb3V0cHV0Q2hhbm5lbHMsIG91dElkKTtcbiAgcmV0dXJuIG91dDtcbn1cblxuZXhwb3J0IGNvbnN0IGNvbnYyRENvbmZpZzogS2VybmVsQ29uZmlnID0ge1xuICBrZXJuZWxOYW1lOiBDb252MkQsXG4gIGJhY2tlbmROYW1lOiAnd2FzbScsXG4gIHNldHVwRnVuYzogc2V0dXAsXG4gIGtlcm5lbEZ1bmM6IGNvbnYyZCBhcyB7fSBhcyBLZXJuZWxGdW5jXG59O1xuIl19