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
import './flags_wasm';
import { DataStorage, deprecationWarn, engine, env, KernelBackend, util } from '@tensorflow/tfjs-core';
import * as wasmFactoryThreadedSimd_import from '../wasm-out/tfjs-backend-wasm-threaded-simd.js';
// @ts-ignore
import { wasmWorkerContents } from '../wasm-out/tfjs-backend-wasm-threaded-simd.worker.js';
import * as wasmFactory_import from '../wasm-out/tfjs-backend-wasm.js';
// This workaround is required for importing in Node.js without using
// the node bundle (for testing). This would not be necessary if we
// flipped esModuleInterop to true, but we likely can't do that since
// google3 does not use it.
const wasmFactoryThreadedSimd = (wasmFactoryThreadedSimd_import.default
    || wasmFactoryThreadedSimd_import);
const wasmFactory = (wasmFactory_import.default
    || wasmFactory_import);
export class BackendWasm extends KernelBackend {
    constructor(wasm) {
        super();
        this.wasm = wasm;
        // 0 is reserved for null data ids.
        this.dataIdNextNumber = 1;
        this.wasm.tfjs.initWithThreadsCount(threadsCount);
        actualThreadsCount = this.wasm.tfjs.getThreadsCount();
        this.dataIdMap = new DataStorage(this, engine());
    }
    write(values, shape, dtype) {
        const dataId = { id: this.dataIdNextNumber++ };
        this.move(dataId, values, shape, dtype, 1);
        return dataId;
    }
    numDataIds() {
        return this.dataIdMap.numDataIds();
    }
    async time(f) {
        const start = util.now();
        f();
        const kernelMs = util.now() - start;
        return { kernelMs };
    }
    move(dataId, values, shape, dtype, refCount) {
        const id = this.dataIdNextNumber++;
        if (dtype === 'string') {
            const stringBytes = values;
            this.dataIdMap.set(dataId, { id, stringBytes, shape, dtype, memoryOffset: null, refCount });
            return;
        }
        const size = util.sizeFromShape(shape);
        const numBytes = size * util.bytesPerElement(dtype);
        const memoryOffset = this.wasm._malloc(numBytes);
        this.dataIdMap.set(dataId, { id, memoryOffset, shape, dtype, refCount });
        this.wasm.tfjs.registerTensor(id, size, memoryOffset);
        if (values != null) {
            this.wasm.HEAPU8.set(new Uint8Array(values.buffer, values.byteOffset, numBytes), memoryOffset);
        }
    }
    async read(dataId) {
        return this.readSync(dataId);
    }
    readSync(dataId, start, end) {
        const { memoryOffset, dtype, shape, stringBytes } = this.dataIdMap.get(dataId);
        if (dtype === 'string') {
            // Slice all elements.
            if ((start == null || start === 0) &&
                (end == null || end >= stringBytes.length)) {
                return stringBytes;
            }
            return stringBytes.slice(start, end);
        }
        start = start || 0;
        end = end || util.sizeFromShape(shape);
        const bytesPerElement = util.bytesPerElement(dtype);
        const bytes = this.wasm.HEAPU8.slice(memoryOffset + start * bytesPerElement, memoryOffset + end * bytesPerElement);
        return typedArrayFromBuffer(bytes.buffer, dtype);
    }
    /**
     * Dispose the memory if the dataId has 0 refCount. Return true if the memory
     * is released, false otherwise.
     * @param dataId
     * @oaram force Optional, remove the data regardless of refCount
     */
    disposeData(dataId, force = false) {
        if (this.dataIdMap.has(dataId)) {
            const data = this.dataIdMap.get(dataId);
            data.refCount--;
            if (!force && data.refCount > 0) {
                return false;
            }
            this.wasm._free(data.memoryOffset);
            this.wasm.tfjs.disposeData(data.id);
            this.dataIdMap.delete(dataId);
        }
        return true;
    }
    /** Return refCount of a `TensorData`. */
    refCount(dataId) {
        if (this.dataIdMap.has(dataId)) {
            const tensorData = this.dataIdMap.get(dataId);
            return tensorData.refCount;
        }
        return 0;
    }
    incRef(dataId) {
        const data = this.dataIdMap.get(dataId);
        if (data != null) {
            data.refCount++;
        }
    }
    floatPrecision() {
        return 32;
    }
    // Returns the memory offset of a tensor. Useful for debugging and unit
    // testing.
    getMemoryOffset(dataId) {
        return this.dataIdMap.get(dataId).memoryOffset;
    }
    dispose() {
        this.wasm.tfjs.dispose();
        if ('PThread' in this.wasm) {
            this.wasm.PThread.terminateAllThreads();
        }
        this.wasm = null;
    }
    memory() {
        return { unreliable: false };
    }
    /**
     * Make a tensor info for the output of an op. If `memoryOffset` is not
     * present, this method allocates memory on the WASM heap. If `memoryOffset`
     * is present, the memory was allocated elsewhere (in c++) and we just record
     * the pointer where that memory lives.
     */
    makeOutput(shape, dtype, memoryOffset) {
        let dataId;
        if (memoryOffset == null) {
            dataId = this.write(null /* values */, shape, dtype);
        }
        else {
            const id = this.dataIdNextNumber++;
            dataId = { id };
            this.dataIdMap.set(dataId, { id, memoryOffset, shape, dtype, refCount: 1 });
            const size = util.sizeFromShape(shape);
            this.wasm.tfjs.registerTensor(id, size, memoryOffset);
        }
        return { dataId, shape, dtype };
    }
    typedArrayFromHeap({ shape, dtype, dataId }) {
        const buffer = this.wasm.HEAPU8.buffer;
        const { memoryOffset } = this.dataIdMap.get(dataId);
        const size = util.sizeFromShape(shape);
        switch (dtype) {
            case 'float32':
                return new Float32Array(buffer, memoryOffset, size);
            case 'int32':
                return new Int32Array(buffer, memoryOffset, size);
            case 'bool':
                return new Uint8Array(buffer, memoryOffset, size);
            default:
                throw new Error(`Unknown dtype ${dtype}`);
        }
    }
}
function createInstantiateWasmFunc(path) {
    // this will be replace by rollup plugin patchWechatWebAssembly in
    // minprogram's output.
    // tslint:disable-next-line:no-any
    return (imports, callback) => {
        util.fetch(path, { credentials: 'same-origin' }).then((response) => {
            if (!response['ok']) {
                imports.env.a(`failed to load wasm binary file at '${path}'`);
            }
            response.arrayBuffer().then(binary => {
                WebAssembly.instantiate(binary, imports).then(output => {
                    callback(output.instance, output.module);
                });
            });
        });
        return {};
    };
}
/**
 * Returns the path of the WASM binary.
 * @param simdSupported whether SIMD is supported
 * @param threadsSupported whether multithreading is supported
 * @param wasmModuleFolder the directory containing the WASM binaries.
 */
function getPathToWasmBinary(simdSupported, threadsSupported, wasmModuleFolder) {
    if (wasmPath != null) {
        // If wasmPath is defined, the user has supplied a full path to
        // the vanilla .wasm binary.
        return wasmPath;
    }
    let path = 'tfjs-backend-wasm.wasm';
    if (simdSupported && threadsSupported) {
        path = 'tfjs-backend-wasm-threaded-simd.wasm';
    }
    else if (simdSupported) {
        path = 'tfjs-backend-wasm-simd.wasm';
    }
    if (wasmFileMap != null) {
        if (wasmFileMap[path] != null) {
            return wasmFileMap[path];
        }
    }
    return wasmModuleFolder + path;
}
/**
 * Initializes the wasm module and creates the js <--> wasm bridge.
 *
 * NOTE: We wrap the wasm module in a object with property 'wasm' instead of
 * returning Promise<BackendWasmModule> to avoid freezing Chrome (last tested
 * in Chrome 76).
 */
export async function init() {
    const [simdSupported, threadsSupported] = await Promise.all([
        env().getAsync('WASM_HAS_SIMD_SUPPORT'),
        env().getAsync('WASM_HAS_MULTITHREAD_SUPPORT')
    ]);
    return new Promise((resolve, reject) => {
        const factoryConfig = {};
        /**
         * This function overrides the Emscripten module locateFile utility.
         * @param path The relative path to the file that needs to be loaded.
         * @param prefix The path to the main JavaScript file's directory.
         */
        factoryConfig.locateFile = (path, prefix) => {
            if (path.endsWith('.worker.js')) {
                // Escape '\n' because Blob will turn it into a newline.
                // There should be a setting for this, but 'endings: "native"' does
                // not seem to work.
                const response = wasmWorkerContents.replace(/\n/g, '\\n');
                const blob = new Blob([response], { type: 'application/javascript' });
                return URL.createObjectURL(blob);
            }
            if (path.endsWith('.wasm')) {
                return getPathToWasmBinary(simdSupported, threadsSupported, wasmPathPrefix != null ? wasmPathPrefix : prefix);
            }
            return prefix + path;
        };
        // Use the instantiateWasm override when system fetch is not available.
        // Reference:
        // https://github.com/emscripten-core/emscripten/blob/2bca083cbbd5a4133db61fbd74d04f7feecfa907/tests/manual_wasm_instantiate.html#L170
        if (customFetch) {
            factoryConfig.instantiateWasm =
                createInstantiateWasmFunc(getPathToWasmBinary(simdSupported, threadsSupported, wasmPathPrefix != null ? wasmPathPrefix : ''));
        }
        let initialized = false;
        factoryConfig.onAbort = () => {
            if (initialized) {
                // Emscripten already called console.warn so no need to double log.
                return;
            }
            if (initAborted) {
                // Emscripten calls `onAbort` twice, resulting in double error
                // messages.
                return;
            }
            initAborted = true;
            const rejectMsg = 'Make sure the server can serve the `.wasm` file relative to the ' +
                'bundled js file. For more details see https://github.com/tensorflow/tfjs/blob/master/tfjs-backend-wasm/README.md#using-bundlers';
            reject({ message: rejectMsg });
        };
        let wasm;
        // If `wasmPath` has been defined we must initialize the vanilla module.
        if (threadsSupported && simdSupported && wasmPath == null) {
            factoryConfig.mainScriptUrlOrBlob = new Blob([`var WasmBackendModuleThreadedSimd = ` +
                    wasmFactoryThreadedSimd.toString()], { type: 'text/javascript' });
            wasm = wasmFactoryThreadedSimd(factoryConfig);
        }
        else {
            // The wasmFactory works for both vanilla and SIMD binaries.
            wasm = wasmFactory(factoryConfig);
        }
        // The `wasm` promise will resolve to the WASM module created by
        // the factory, but it might have had errors during creation. Most
        // errors are caught by the onAbort callback defined above.
        // However, some errors, such as those occurring from a
        // failed fetch, result in this promise being rejected. These are
        // caught and re-rejected below.
        wasm.then((module) => {
            initialized = true;
            initAborted = false;
            const voidReturnType = null;
            // Using the tfjs namespace to avoid conflict with emscripten's API.
            module.tfjs = {
                init: module.cwrap('init', null, []),
                initWithThreadsCount: module.cwrap('init_with_threads_count', null, ['number']),
                getThreadsCount: module.cwrap('get_threads_count', 'number', []),
                registerTensor: module.cwrap('register_tensor', null, [
                    'number',
                    'number',
                    'number', // memoryOffset
                ]),
                disposeData: module.cwrap('dispose_data', voidReturnType, ['number']),
                dispose: module.cwrap('dispose', voidReturnType, []),
            };
            resolve({ wasm: module });
        }).catch(reject);
    });
}
function typedArrayFromBuffer(buffer, dtype) {
    switch (dtype) {
        case 'float32':
            return new Float32Array(buffer);
        case 'int32':
            return new Int32Array(buffer);
        case 'bool':
            return new Uint8Array(buffer);
        default:
            throw new Error(`Unknown dtype ${dtype}`);
    }
}
const wasmBinaryNames = [
    'tfjs-backend-wasm.wasm', 'tfjs-backend-wasm-simd.wasm',
    'tfjs-backend-wasm-threaded-simd.wasm'
];
let wasmPath = null;
let wasmPathPrefix = null;
let wasmFileMap = {};
let initAborted = false;
let customFetch = false;
/**
 * @deprecated Use `setWasmPaths` instead.
 * Sets the path to the `.wasm` file which will be fetched when the wasm
 * backend is initialized. See
 * https://github.com/tensorflow/tfjs/blob/master/tfjs-backend-wasm/README.md#using-bundlers
 * for more details.
 * @param path wasm file path or url
 * @param usePlatformFetch optional boolean to use platform fetch to download
 *     the wasm file, default to false.
 *
 * @doc {heading: 'Environment', namespace: 'wasm'}
 */
export function setWasmPath(path, usePlatformFetch = false) {
    deprecationWarn('setWasmPath has been deprecated in favor of setWasmPaths and' +
        ' will be removed in a future release.');
    if (initAborted) {
        throw new Error('The WASM backend was already initialized. Make sure you call ' +
            '`setWasmPath()` before you call `tf.setBackend()` or `tf.ready()`');
    }
    wasmPath = path;
    customFetch = usePlatformFetch;
}
/**
 * Configures the locations of the WASM binaries.
 *
 * ```js
 * setWasmPaths({
 *  'tfjs-backend-wasm.wasm': 'renamed.wasm',
 *  'tfjs-backend-wasm-simd.wasm': 'renamed-simd.wasm',
 *  'tfjs-backend-wasm-threaded-simd.wasm': 'renamed-threaded-simd.wasm'
 * });
 * tf.setBackend('wasm');
 * ```
 *
 * @param prefixOrFileMap This can be either a string or object:
 *  - (string) The path to the directory where the WASM binaries are located.
 *     Note that this prefix will be used to load each binary (vanilla,
 *     SIMD-enabled, threading-enabled, etc.).
 *  - (object) Mapping from names of WASM binaries to custom
 *     full paths specifying the locations of those binaries. This is useful if
 *     your WASM binaries are not all located in the same directory, or if your
 *     WASM binaries have been renamed.
 * @param usePlatformFetch optional boolean to use platform fetch to download
 *     the wasm file, default to false.
 *
 * @doc {heading: 'Environment', namespace: 'wasm'}
 */
export function setWasmPaths(prefixOrFileMap, usePlatformFetch = false) {
    if (initAborted) {
        throw new Error('The WASM backend was already initialized. Make sure you call ' +
            '`setWasmPaths()` before you call `tf.setBackend()` or ' +
            '`tf.ready()`');
    }
    if (typeof prefixOrFileMap === 'string') {
        wasmPathPrefix = prefixOrFileMap;
    }
    else {
        wasmFileMap = prefixOrFileMap;
        const missingPaths = wasmBinaryNames.filter(name => wasmFileMap[name] == null);
        if (missingPaths.length > 0) {
            throw new Error(`There were no entries found for the following binaries: ` +
                `${missingPaths.join(',')}. Please either call setWasmPaths with a ` +
                `map providing a path for each binary, or with a string indicating ` +
                `the directory where all the binaries can be found.`);
        }
    }
    customFetch = usePlatformFetch;
}
/** Used in unit tests. */
export function resetWasmPath() {
    wasmPath = null;
    wasmPathPrefix = null;
    wasmFileMap = {};
    customFetch = false;
    initAborted = false;
}
let threadsCount = -1;
let actualThreadsCount = -1;
/**
 * Sets the number of threads that will be used by XNNPACK to create
 * threadpool (default to the number of logical CPU cores).
 *
 * This must be called before calling `tf.setBackend('wasm')`.
 */
export function setThreadsCount(numThreads) {
    threadsCount = numThreads;
}
/**
 * Gets the actual threads count that is used by XNNPACK.
 *
 * It is set after the backend is intialized.
 */
export function getThreadsCount() {
    if (actualThreadsCount === -1) {
        throw new Error(`WASM backend not initialized.`);
    }
    return actualThreadsCount;
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiYmFja2VuZF93YXNtLmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vdGZqcy1iYWNrZW5kLXdhc20vc3JjL2JhY2tlbmRfd2FzbS50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7Ozs7Ozs7O0dBZUc7QUFDSCxPQUFPLGNBQWMsQ0FBQztBQUV0QixPQUFPLEVBQWtDLFdBQVcsRUFBWSxlQUFlLEVBQUUsTUFBTSxFQUFFLEdBQUcsRUFBRSxhQUFhLEVBQWMsSUFBSSxFQUFDLE1BQU0sdUJBQXVCLENBQUM7QUFJNUosT0FBUSxLQUFLLDhCQUE4QixNQUFNLGdEQUFnRCxDQUFDO0FBQ2xHLGFBQWE7QUFDYixPQUFPLEVBQUMsa0JBQWtCLEVBQUMsTUFBTSx1REFBdUQsQ0FBQztBQUN6RixPQUFPLEtBQUssa0JBQWtCLE1BQU0sa0NBQWtDLENBQUM7QUFFdkUscUVBQXFFO0FBQ3JFLG1FQUFtRTtBQUNuRSxxRUFBcUU7QUFDckUsMkJBQTJCO0FBQzNCLE1BQU0sdUJBQXVCLEdBQUcsQ0FBQyw4QkFBOEIsQ0FBQyxPQUFPO09BQ2xFLDhCQUE4QixDQUNVLENBQUM7QUFDOUMsTUFBTSxXQUFXLEdBQUcsQ0FBQyxrQkFBa0IsQ0FBQyxPQUFPO09BQzFDLGtCQUFrQixDQUFzQyxDQUFDO0FBYzlELE1BQU0sT0FBTyxXQUFZLFNBQVEsYUFBYTtJQUs1QyxZQUFtQixJQUFxRDtRQUN0RSxLQUFLLEVBQUUsQ0FBQztRQURTLFNBQUksR0FBSixJQUFJLENBQWlEO1FBSnhFLG1DQUFtQztRQUMzQixxQkFBZ0IsR0FBRyxDQUFDLENBQUM7UUFLM0IsSUFBSSxDQUFDLElBQUksQ0FBQyxJQUFJLENBQUMsb0JBQW9CLENBQUMsWUFBWSxDQUFDLENBQUM7UUFDbEQsa0JBQWtCLEdBQUcsSUFBSSxDQUFDLElBQUksQ0FBQyxJQUFJLENBQUMsZUFBZSxFQUFFLENBQUM7UUFDdEQsSUFBSSxDQUFDLFNBQVMsR0FBRyxJQUFJLFdB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R0FBRyxJQUFJLENBQUM7SUFDdEIsV0FBVyxHQUFHLEVBQUUsQ0FBQztJQUNqQixXQUFXLEdBQUcsS0FBSyxDQUFDO0lBQ3BCLFdBQVcsR0FBRyxLQUFLLENBQUM7QUFDdEIsQ0FBQztBQUVELElBQUksWUFBWSxHQUFHLENBQUMsQ0FBQyxDQUFDO0FBQ3RCLElBQUksa0JBQWtCLEdBQUcsQ0FBQyxDQUFDLENBQUM7QUFFNUI7Ozs7O0dBS0c7QUFDSCxNQUFNLFVBQVUsZUFBZSxDQUFDLFVBQWtCO0lBQ2hELFlBQVksR0FBRyxVQUFVLENBQUM7QUFDNUIsQ0FBQztBQUVEOzs7O0dBSUc7QUFDSCxNQUFNLFVBQVUsZUFBZTtJQUM3QixJQUFJLGtCQUFrQixLQUFLLENBQUMsQ0FBQyxFQUFFO1FBQzdCLE1BQU0sSUFBSSxLQUFLLENBQUMsK0JBQStCLENBQUMsQ0FBQztLQUNsRDtJQUNELE9BQU8sa0JBQWtCLENBQUM7QUFDNUIsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDE5IEdvb2dsZSBMTEMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cbmltcG9ydCAnLi9mbGFnc193YXNtJztcblxuaW1wb3J0IHtiYWNrZW5kX3V0aWwsIEJhY2tlbmRUaW1pbmdJbmZvLCBEYXRhU3RvcmFnZSwgRGF0YVR5cGUsIGRlcHJlY2F0aW9uV2FybiwgZW5naW5lLCBlbnYsIEtlcm5lbEJhY2tlbmQsIFRlbnNvckluZm8sIHV0aWx9IGZyb20gJ0B0ZW5zb3JmbG93L3RmanMtY29yZSc7XG5cbmltcG9ydCB7QmFja2VuZFdhc21Nb2R1bGUsIFdhc21GYWN0b3J5Q29uZmlnfSBmcm9tICcuLi93YXNtLW91dC90ZmpzLWJhY2tlbmQtd2FzbSc7XG5pbXBvcnQge0JhY2tlbmRXYXNtVGhyZWFkZWRTaW1kTW9kdWxlfSBmcm9tICcuLi93YXNtLW91dC90ZmpzLWJhY2tlbmQtd2FzbS10aHJlYWRlZC1zaW1kJztcbmltcG9ydCAgKiBhcyB3YXNtRmFjdG9yeVRocmVhZGVkU2ltZF9pbXBvcnQgZnJvbSAnLi4vd2FzbS1vdXQvdGZqcy1iYWNrZW5kLXdhc20tdGhyZWFkZWQtc2ltZC5qcyc7XG4vLyBAdHMtaWdub3JlXG5pbXBvcnQge3dhc21Xb3JrZXJDb250ZW50c30gZnJvbSAnLi4vd2FzbS1vdXQvdGZqcy1iYWNrZW5kLXdhc20tdGhyZWFkZWQtc2ltZC53b3JrZXIuanMnO1xuaW1wb3J0ICogYXMgd2FzbUZhY3RvcnlfaW1wb3J0IGZyb20gJy4uL3dhc20tb3V0L3RmanMtYmFja2VuZC13YXNtLmpzJztcblxuLy8gVGhpcyB3b3JrYXJvdW5kIGlzIHJlcXVpcmVkIGZvciBpbXBvcnRpbmcgaW4gTm9kZS5qcyB3aXRob3V0IHVzaW5nXG4vLyB0aGUgbm9kZSBidW5kbGUgKGZvciB0ZXN0aW5nKS4gVGhpcyB3b3VsZCBub3QgYmUgbmVjZXNzYXJ5IGlmIHdlXG4vLyBmbGlwcGVkIGVzTW9kdWxlSW50ZXJvcCB0byB0cnVlLCBidXQgd2UgbGlrZWx5IGNhbid0IGRvIHRoYXQgc2luY2Vcbi8vIGdvb2dsZTMgZG9lcyBub3QgdXNlIGl0LlxuY29uc3Qgd2FzbUZhY3RvcnlUaHJlYWRlZFNpbWQgPSAod2FzbUZhY3RvcnlUaHJlYWRlZFNpbWRfaW1wb3J0LmRlZmF1bHRcbiAgfHwgd2FzbUZhY3RvcnlUaHJlYWRlZFNpbWRfaW1wb3J0KSBhc1xudHlwZW9mIHdhc21GYWN0b3J5VGhyZWFkZWRTaW1kX2ltcG9ydC5kZWZhdWx0O1xuY29uc3Qgd2FzbUZhY3RvcnkgPSAod2FzbUZhY3RvcnlfaW1wb3J0LmRlZmF1bHRcbiAgfHwgd2FzbUZhY3RvcnlfaW1wb3J0KSBhcyB0eXBlb2Ygd2FzbUZhY3RvcnlfaW1wb3J0LmRlZmF1bHQ7XG5cbmludGVyZmFjZSBUZW5zb3JEYXRhIHtcbiAgaWQ6IG51bWJlcjtcbiAgbWVtb3J5T2Zmc2V0OiBudW1iZXI7XG4gIHNoYXBlOiBudW1iZXJbXTtcbiAgZHR5cGU6IERhdGFUeXBlO1xuICByZWZDb3VudDogbnVtYmVyO1xuICAvKiogT25seSB1c2VkIGZvciBzdHJpbmcgdGVuc29ycywgc3RvcmluZyBlbmNvZGVkIGJ5dGVzLiAqL1xuICBzdHJpbmdCeXRlcz86IFVpbnQ4QXJyYXlbXTtcbn1cblxuZXhwb3J0IHR5cGUgRGF0YUlkID0gb2JqZWN0OyAgLy8gb2JqZWN0IGluc3RlYWQgb2Yge30gdG8gZm9yY2Ugbm9uLXByaW1pdGl2ZS5cblxuZXhwb3J0IGNsYXNzIEJhY2tlbmRXYXNtIGV4dGVuZHMgS2VybmVsQmFja2VuZCB7XG4gIC8vIDAgaXMgcmVzZXJ2ZWQgZm9yIG51bGwgZGF0YSBpZHMuXG4gIHByaXZhdGUgZGF0YUlkTmV4dE51bWJlciA9IDE7XG4gIGRhdGFJZE1hcDogRGF0YVN0b3JhZ2U8VGVuc29yRGF0YT47XG5cbiAgY29uc3RydWN0b3IocHVibGljIHdhc206IEJhY2tlbmRXYXNtTW9kdWxlfEJhY2tlbmRXYXNtVGhyZWFkZWRTaW1kTW9kdWxlKSB7XG4gICAgc3VwZXIoKTtcbiAgICB0aGlzLndhc20udGZqcy5pbml0V2l0aFRocmVhZHNDb3VudCh0aHJlYWRzQ291bnQpO1xuICAgIGFjdHVhbFRocmVhZHNDb3VudCA9IHRoaXMud2FzbS50ZmpzLmdldFRocmVhZHNDb3VudCgpO1xuICAgIHRoaXMuZGF0YUlkTWFwID0gbmV3IERhdGFTdG9yYWdlKHRoaXMsIGVuZ2luZSgpKTtcbiAgfVxuXG4gIHdyaXRlKHZhbHVlczogYmFja2VuZF91dGlsLkJhY2tlbmRWYWx1ZXMsIHNoYXBlOiBudW1iZXJbXSwgZHR5cGU6IERhdGFUeXBlKTpcbiAgICAgIERhdGFJZCB7XG4gICAgY29uc3QgZGF0YUlkID0ge2lkOiB0aGlzLmRhdGFJZE5leHROdW1iZXIrK307XG4gICAgdGhpcy5tb3ZlKGRhdGFJZCwgdmFsdWVzLCBzaGFwZSwgZHR5cGUsIDEpO1xuICAgIHJldHVybiBkYXRhSWQ7XG4gIH1cblxuICBudW1EYXRhSWRzKCk6IG51bWJlciB7XG4gICAgcmV0dXJuIHRoaXMuZGF0YUlkTWFwLm51bURhdGFJZHMoKTtcbiAgfVxuXG4gIGFzeW5jIHRpbWUoZjogKCkgPT4gdm9pZCk6IFByb21pc2U8QmFja2VuZFRpbWluZ0luZm8+IHtcbiAgICBjb25zdCBzdGFydCA9IHV0aWwubm93KCk7XG4gICAgZigpO1xuICAgIGNvbnN0IGtlcm5lbE1zID0gdXRpbC5ub3coKSAtIHN0YXJ0O1xuICAgIHJldHVybiB7a2VybmVsTXN9O1xuICB9XG5cbiAgbW92ZShcbiAgICAgIGRhdGFJZDogRGF0YUlkLCB2YWx1ZXM6IGJhY2tlbmRfdXRpbC5CYWNrZW5kVmFsdWVzLCBzaGFwZTogbnVtYmVyW10sXG4gICAgICBkdHlwZTogRGF0YVR5cGUsIHJlZkNvdW50OiBudW1iZXIpOiB2b2lkIHtcbiAgICBjb25zdCBpZCA9IHRoaXMuZGF0YUlkTmV4dE51bWJlcisrO1xuICAgIGlmIChkdHlwZSA9PT0gJ3N0cmluZycpIHtcbiAgICAgIGNvbnN0IHN0cmluZ0J5dGVzID0gdmFsdWVzIGFzIFVpbnQ4QXJyYXlbXTtcbiAgICAgIHRoaXMuZGF0YUlkTWFwLnNldChcbiAgICAgICAgICBkYXRhSWQsXG4gICAgICAgICAge2lkLCBzdHJpbmdCeXRlcywgc2hhcGUsIGR0eXBlLCBtZW1vcnlPZmZzZXQ6IG51bGwsIHJlZkNvdW50fSk7XG4gICAgICByZXR1cm47XG4gICAgfVxuXG4gICAgY29uc3Qgc2l6ZSA9IHV0aWwuc2l6ZUZyb21TaGFwZShzaGFwZSk7XG4gICAgY29uc3QgbnVtQnl0ZXMgPSBzaXplICogdXRpbC5ieXRlc1BlckVsZW1lbnQoZHR5cGUpO1xuICAgIGNvbnN0IG1lbW9yeU9mZnNldCA9IHRoaXMud2FzbS5fbWFsbG9jKG51bUJ5dGVzKTtcblxuICAgIHRoaXMuZGF0YUlkTWFwLnNldChkYXRhSWQsIHtpZCwgbWVtb3J5T2Zmc2V0LCBzaGFwZSwgZHR5cGUsIHJlZkNvdW50fSk7XG5cbiAgICB0aGlzLndhc20udGZqcy5yZWdpc3RlclRlbnNvcihpZCwgc2l6ZSwgbWVtb3J5T2Zmc2V0KTtcblxuICAgIGlmICh2YWx1ZXMgIT0gbnVsbCkge1xuICAgICAgdGhpcy53YXNtLkhFQVBVOC5zZXQoXG4gICAgICAgICAgbmV3IFVpbnQ4QXJyYXkoXG4gICAgICAgICAgICAgICh2YWx1ZXMgYXMgYmFja2VuZF91dGlsLlR5cGVkQXJyYXkpLmJ1ZmZlcixcbiAgICAgICAgICAgICAgKHZhbHVlcyBhcyBiYWNrZW5kX3V0aWwuVHlwZWRBcnJheSkuYnl0ZU9mZnNldCwgbnVtQnl0ZXMpLFxuICAgICAgICAgIG1lbW9yeU9mZnNldCk7XG4gICAgfVxuICB9XG5cbiAgYXN5bmMgcmVhZChkYXRhSWQ6IERhdGFJZCk6IFByb21pc2U8YmFja2VuZF91dGlsLkJhY2tlbmRWYWx1ZXM+IHtcbiAgICByZXR1cm4gdGhpcy5yZWFkU3luYyhkYXRhSWQpO1xuICB9XG5cbiAgcmVhZFN5bmMoZGF0YUlkOiBEYXRhSWQsIHN0YXJ0PzogbnVtYmVyLCBlbmQ/OiBudW1iZXIpOlxuICAgICAgYmFja2VuZF91dGlsLkJhY2tlbmRWYWx1ZXMge1xuICAgIGNvbnN0IHttZW1vcnlPZmZzZXQsIGR0eXBlLCBzaGFwZSwgc3RyaW5nQnl0ZXN9ID1cbiAgICAgICAgdGhpcy5kYXRhSWRNYXAuZ2V0KGRhdGFJZCk7XG4gICAgaWYgKGR0eXBlID09PSAnc3RyaW5nJykge1xuICAgICAgLy8gU2xpY2UgYWxsIGVsZW1lbnRzLlxuICAgICAgaWYgKChzdGFydCA9PSBudWxsIHx8IHN0YXJ0ID09PSAwKSAmJlxuICAgICAgICAgIChlbmQgPT0gbnVsbCB8fCBlbmQgPj0gc3RyaW5nQnl0ZXMubGVuZ3RoKSkge1xuICAgICAgICByZXR1cm4gc3RyaW5nQnl0ZXM7XG4gICAgICB9XG4gICAgICByZXR1cm4gc3RyaW5nQnl0ZXMuc2xpY2Uoc3RhcnQsIGVuZCk7XG4gICAgfVxuICAgIHN0YXJ0ID0gc3RhcnQgfHwgMDtcbiAgICBlbmQgPSBlbmQgfHwgdXRpbC5zaXplRnJvbVNoYXBlKHNoYXBlKTtcbiAgICBjb25zdCBieXRlc1BlckVsZW1lbnQgPSB1dGlsLmJ5dGVzUGVyRWxlbWVudChkdHlwZSk7XG4gICAgY29uc3QgYnl0ZXMgPSB0aGlzLndhc20uSEVBUFU4LnNsaWNlKFxuICAgICAgICBtZW1vcnlPZmZzZXQgKyBzdGFydCAqIGJ5dGVzUGVyRWxlbWVudCxcbiAgICAgICAgbWVtb3J5T2Zmc2V0ICsgZW5kICogYnl0ZXNQZXJFbGVtZW50KTtcbiAgICByZXR1cm4gdHlwZWRBcnJheUZyb21CdWZmZXIoYnl0ZXMuYnVmZmVyLCBkdHlwZSk7XG4gIH1cblxuICAvKipcbiAgICogRGlzcG9zZSB0aGUgbWVtb3J5IGlmIHRoZSBkYXRhSWQgaGFzIDAgcmVmQ291bnQuIFJldHVybiB0cnVlIGlmIHRoZSBtZW1vcnlcbiAgICogaXMgcmVsZWFzZWQsIGZhbHNlIG90aGVyd2lzZS5cbiAgICogQHBhcmFtIGRhdGFJZFxuICAgKiBAb2FyYW0gZm9yY2UgT3B0aW9uYWwsIHJlbW92ZSB0aGUgZGF0YSByZWdhcmRsZXNzIG9mIHJlZkNvdW50XG4gICAqL1xuICBkaXNwb3NlRGF0YShkYXRhSWQ6IERhdGFJZCwgZm9yY2UgPSBmYWxzZSk6IGJvb2xlYW4ge1xuICAgIGlmICh0aGlzLmRhdGFJZE1hcC5oYXMoZGF0YUlkKSkge1xuICAgICAgY29uc3QgZGF0YSA9IHRoaXMuZGF0YUlkTWFwLmdldChkYXRhSWQpO1xuICAgICAgZGF0YS5yZWZDb3VudC0tO1xuICAgICAgaWYgKCFmb3JjZSAmJiBkYXRhLnJlZkNvdW50ID4gMCkge1xuICAgICAgICByZXR1cm4gZmFsc2U7XG4gICAgICB9XG5cbiAgICAgIHRoaXMud2FzbS5fZnJlZShkYXRhLm1lbW9yeU9mZnNldCk7XG4gICAgICB0aGlzLndhc20udGZqcy5kaXNwb3NlRGF0YShkYXRhLmlkKTtcbiAgICAgIHRoaXMuZGF0YUlkTWFwLmRlbGV0ZShkYXRhSWQpO1xuICAgIH1cbiAgICByZXR1cm4gdHJ1ZTtcbiAgfVxuXG4gIC8qKiBSZXR1cm4gcmVmQ291bnQgb2YgYSBgVGVuc29yRGF0YWAuICovXG4gIHJlZkNvdW50KGRhdGFJZDogRGF0YUlkKTogbnVtYmVyIHtcbiAgICBpZiAodGhpcy5kYXRhSWRNYXAuaGFzKGRhdGFJZCkpIHtcbiAgICAgIGNvbnN0IHRlbnNvckRhdGEgPSB0aGlzLmRhdGFJZE1hcC5nZXQoZGF0YUlkKTtcbiAgICAgIHJldHVybiB0ZW5zb3JEYXRhLnJlZkNvdW50O1xuICAgIH1cbiAgICByZXR1cm4gMDtcbiAgfVxuXG4gIGluY1JlZihkYXRhSWQ6IERhdGFJZCkge1xuICAgIGNvbnN0IGRhdGEgPSB0aGlzLmRhdGFJZE1hcC5nZXQoZGF0YUlkKTtcbiAgICBpZiAoZGF0YSAhPSBudWxsKSB7XG4gICAgICBkYXRhLnJlZkNvdW50Kys7XG4gICAgfVxuICB9XG5cbiAgZmxvYXRQcmVjaXNpb24oKTogMzIge1xuICAgIHJldHVybiAzMjtcbiAgfVxuXG4gIC8vIFJldHVybnMgdGhlIG1lbW9yeSBvZmZzZXQgb2YgYSB0ZW5zb3IuIFVzZWZ1bCBmb3IgZGVidWdnaW5nIGFuZCB1bml0XG4gIC8vIHRlc3RpbmcuXG4gIGdldE1lbW9yeU9mZnNldChkYXRhSWQ6IERhdGFJZCk6IG51bWJlciB7XG4gICAgcmV0dXJuIHRoaXMuZGF0YUlkTWFwLmdldChkYXRhSWQpLm1lbW9yeU9mZnNldDtcbiAgfVxuXG4gIGRpc3Bvc2UoKSB7XG4gICAgdGhpcy53YXNtLnRmanMuZGlzcG9zZSgpO1xuICAgIGlmICgnUFRocmVhZCcgaW4gdGhpcy53YXNtKSB7XG4gICAgICB0aGlzLndhc20uUFRocmVhZC50ZXJtaW5hdGVBbGxUaHJlYWRzKCk7XG4gICAgfVxuICAgIHRoaXMud2FzbSA9IG51bGw7XG4gIH1cblxuICBtZW1vcnkoKSB7XG4gICAgcmV0dXJuIHt1bnJlbGlhYmxlOiBmYWxzZX07XG4gIH1cblxuICAvKipcbiAgICogTWFrZSBhIHRlbnNvciBpbmZvIGZvciB0aGUgb3V0cHV0IG9mIGFuIG9wLiBJZiBgbWVtb3J5T2Zmc2V0YCBpcyBub3RcbiAgICogcHJlc2VudCwgdGhpcyBtZXRob2QgYWxsb2NhdGVzIG1lbW9yeSBvbiB0aGUgV0FTTSBoZWFwLiBJZiBgbWVtb3J5T2Zmc2V0YFxuICAgKiBpcyBwcmVzZW50LCB0aGUgbWVtb3J5IHdhcyBhbGxvY2F0ZWQgZWxzZXdoZXJlIChpbiBjKyspIGFuZCB3ZSBqdXN0IHJlY29yZFxuICAgKiB0aGUgcG9pbnRlciB3aGVyZSB0aGF0IG1lbW9yeSBsaXZlcy5cbiAgICovXG4gIG1ha2VPdXRwdXQoc2hhcGU6IG51bWJlcltdLCBkdHlwZTogRGF0YVR5cGUsIG1lbW9yeU9mZnNldD86IG51bWJlcik6XG4gICAgICBUZW5zb3JJbmZvIHtcbiAgICBsZXQgZGF0YUlkOiB7fTtcbiAgICBpZiAobWVtb3J5T2Zmc2V0ID09IG51bGwpIHtcbiAgICAgIGRhdGFJZCA9IHRoaXMud3JpdGUobnVsbCAvKiB2YWx1ZXMgKi8sIHNoYXBlLCBkdHlwZSk7XG4gICAgfSBlbHNlIHtcbiAgICAgIGNvbnN0IGlkID0gdGhpcy5kYXRhSWROZXh0TnVtYmVyKys7XG4gICAgICBkYXRhSWQgPSB7aWR9O1xuICAgICAgdGhpcy5kYXRhSWRNYXAuc2V0KGRhdGFJZCwge2lkLCBtZW1vcnlPZmZzZXQsIHNoYXBlLCBkdHlwZSwgcmVmQ291bnQ6IDF9KTtcbiAgICAgIGNvbnN0IHNpemUgPSB1dGlsLnNpemVGcm9tU2hhcGUoc2hhcGUpO1xuICAgICAgdGhpcy53YXNtLnRmanMucmVnaXN0ZXJUZW5zb3IoaWQsIHNpemUsIG1lbW9yeU9mZnNldCk7XG4gICAgfVxuICAgIHJldHVybiB7ZGF0YUlkLCBzaGFwZSwgZHR5cGV9O1xuICB9XG5cbiAgdHlwZWRBcnJheUZyb21IZWFwKHtzaGFwZSwgZHR5cGUsIGRhdGFJZH06IFRlbnNvckluZm8pOlxuICAgICAgYmFja2VuZF91dGlsLlR5cGVkQXJyYXkge1xuICAgIGNvbnN0IGJ1ZmZlciA9IHRoaXMud2FzbS5IRUFQVTguYnVmZmVyO1xuICAgIGNvbnN0IHttZW1vcnlPZmZzZXR9ID0gdGhpcy5kYXRhSWRNYXAuZ2V0KGRhdGFJZCk7XG4gICAgY29uc3Qgc2l6ZSA9IHV0aWwuc2l6ZUZyb21TaGFwZShzaGFwZSk7XG4gICAgc3dpdGNoIChkdHlwZSkge1xuICAgICAgY2FzZSAnZmxvYXQzMic6XG4gICAgICAgIHJldHVybiBuZXcgRmxvYXQzMkFycmF5KGJ1ZmZlciwgbWVtb3J5T2Zmc2V0LCBzaXplKTtcbiAgICAgIGNhc2UgJ2ludDMyJzpcbiAgICAgICAgcmV0dXJuIG5ldyBJbnQzMkFycmF5KGJ1ZmZlciwgbWVtb3J5T2Zmc2V0LCBzaXplKTtcbiAgICAgIGNhc2UgJ2Jvb2wnOlxuICAgICAgICByZXR1cm4gbmV3IFVpbnQ4QXJyYXkoYnVmZmVyLCBtZW1vcnlPZmZzZXQsIHNpemUpO1xuICAgICAgZGVmYXVsdDpcbiAgICAgICAgdGhyb3cgbmV3IEVycm9yKGBVbmtub3duIGR0eXBlICR7ZHR5cGV9YCk7XG4gICAgfVxuICB9XG59XG5cbmZ1bmN0aW9uIGNyZWF0ZUluc3RhbnRpYXRlV2FzbUZ1bmMocGF0aDogc3RyaW5nKSB7XG4gIC8vIHRoaXMgd2lsbCBiZSByZXBsYWNlIGJ5IHJvbGx1cCBwbHVnaW4gcGF0Y2hXZWNoYXRXZWJBc3NlbWJseSBpblxuICAvLyBtaW5wcm9ncmFtJ3Mgb3V0cHV0LlxuICAvLyB0c2xpbnQ6ZGlzYWJsZS1uZXh0LWxpbmU6bm8tYW55XG4gIHJldHVybiAoaW1wb3J0czogYW55LCBjYWxsYmFjazogYW55KSA9PiB7XG4gICAgdXRpbC5mZXRjaChwYXRoLCB7Y3JlZGVudGlhbHM6ICdzYW1lLW9yaWdpbid9KS50aGVuKChyZXNwb25zZSkgPT4ge1xuICAgICAgaWYgKCFyZXNwb25zZVsnb2snXSkge1xuICAgICAgICBpbXBvcnRzLmVudi5hKGBmYWlsZWQgdG8gbG9hZCB3YXNtIGJpbmFyeSBmaWxlIGF0ICcke3BhdGh9J2ApO1xuICAgICAgfVxuICAgICAgcmVzcG9uc2UuYXJyYXlCdWZmZXIoKS50aGVuKGJpbmFyeSA9PiB7XG4gICAgICAgIFdlYkFzc2VtYmx5Lmluc3RhbnRpYXRlKGJpbmFyeSwgaW1wb3J0cykudGhlbihvdXRwdXQgPT4ge1xuICAgICAgICAgIGNhbGxiYWNrKG91dHB1dC5pbnN0YW5jZSwgb3V0cHV0Lm1vZHVsZSk7XG4gICAgICAgIH0pO1xuICAgICAgfSk7XG4gICAgfSk7XG4gICAgcmV0dXJuIHt9O1xuICB9O1xufVxuXG4vKipcbiAqIFJldHVybnMgdGhlIHBhdGggb2YgdGhlIFdBU00gYmluYXJ5LlxuICogQHBhcmFtIHNpbWRTdXBwb3J0ZWQgd2hldGhlciBTSU1EIGlzIHN1cHBvcnRlZFxuICogQHBhcmFtIHRocmVhZHNTdXBwb3J0ZWQgd2hldGhlciBtdWx0aXRocmVhZGluZyBpcyBzdXBwb3J0ZWRcbiAqIEBwYXJhbSB3YXNtTW9kdWxlRm9sZGVyIHRoZSBkaXJlY3RvcnkgY29udGFpbmluZyB0aGUgV0FTTSBiaW5hcmllcy5cbiAqL1xuZnVuY3Rpb24gZ2V0UGF0aFRvV2FzbUJpbmFyeShcbiAgICBzaW1kU3VwcG9ydGVkOiBib29sZWFuLCB0aHJlYWRzU3VwcG9ydGVkOiBib29sZWFuLFxuICAgIHdhc21Nb2R1bGVGb2xkZXI6IHN0cmluZykge1xuICBpZiAod2FzbVBhdGggIT0gbnVsbCkge1xuICAgIC8vIElmIHdhc21QYXRoIGlzIGRlZmluZWQsIHRoZSB1c2VyIGhhcyBzdXBwbGllZCBhIGZ1bGwgcGF0aCB0b1xuICAgIC8vIHRoZSB2YW5pbGxhIC53YXNtIGJpbmFyeS5cbiAgICByZXR1cm4gd2FzbVBhdGg7XG4gIH1cblxuICBsZXQgcGF0aDogV2FzbUJpbmFyeU5hbWUgPSAndGZqcy1iYWNrZW5kLXdhc20ud2FzbSc7XG4gIGlmIChzaW1kU3VwcG9ydGVkICYmIHRocmVhZHNTdXBwb3J0ZWQpIHtcbiAgICBwYXRoID0gJ3RmanMtYmFja2VuZC13YXNtLXRocmVhZGVkLXNpbWQud2FzbSc7XG4gIH0gZWxzZSBpZiAoc2ltZFN1cHBvcnRlZCkge1xuICAgIHBhdGggPSAndGZqcy1iYWNrZW5kLXdhc20tc2ltZC53YXNtJztcbiAgfVxuXG4gIGlmICh3YXNtRmlsZU1hcCAhPSBudWxsKSB7XG4gICAgaWYgKHdhc21GaWxlTWFwW3BhdGhdICE9IG51bGwpIHtcbiAgICAgIHJldHVybiB3YXNtRmlsZU1hcFtwYXRoXTtcbiAgICB9XG4gIH1cblxuICByZXR1cm4gd2FzbU1vZHVsZUZvbGRlciArIHBhdGg7XG59XG5cbi8qKlxuICogSW5pdGlhbGl6ZXMgdGhlIHdhc20gbW9kdWxlIGFuZCBjcmVhdGVzIHRoZSBqcyA8LS0+IHdhc20gYnJpZGdlLlxuICpcbiAqIE5PVEU6IFdlIHdyYXAgdGhlIHdhc20gbW9kdWxlIGluIGEgb2JqZWN0IHdpdGggcHJvcGVydHkgJ3dhc20nIGluc3RlYWQgb2ZcbiAqIHJldHVybmluZyBQcm9taXNlPEJhY2tlbmRXYXNtTW9kdWxlPiB0byBhdm9pZCBmcmVlemluZyBDaHJvbWUgKGxhc3QgdGVzdGVkXG4gKiBpbiBDaHJvbWUgNzYpLlxuICovXG5leHBvcnQgYXN5bmMgZnVuY3Rpb24gaW5pdCgpOiBQcm9taXNlPHt3YXNtOiBCYWNrZW5kV2FzbU1vZHVsZX0+IHtcbiAgY29uc3QgW3NpbWRTdXBwb3J0ZWQsIHRocmVhZHNTdXBwb3J0ZWRdID0gYXdhaXQgUHJvbWlzZS5hbGwoW1xuICAgIGVudigpLmdldEFzeW5jKCdXQVNNX0hBU19TSU1EX1NVUFBPUlQnKSxcbiAgICBlbnYoKS5nZXRBc3luYygnV0FTTV9IQVNfTVVMVElUSFJFQURfU1VQUE9SVCcpXG4gIF0pO1xuXG4gIHJldHVybiBuZXcgUHJvbWlzZSgocmVzb2x2ZSwgcmVqZWN0KSA9PiB7XG4gICAgY29uc3QgZmFjdG9yeUNvbmZpZzogV2FzbUZhY3RvcnlDb25maWcgPSB7fTtcblxuICAgIC8qKlxuICAgICAqIFRoaXMgZnVuY3Rpb24gb3ZlcnJpZGVzIHRoZSBFbXNjcmlwdGVuIG1vZHVsZSBsb2NhdGVGaWxlIHV0aWxpdHkuXG4gICAgICogQHBhcmFtIHBhdGggVGhlIHJlbGF0aXZlIHBhdGggdG8gdGhlIGZpbGUgdGhhdCBuZWVkcyB0byBiZSBsb2FkZWQuXG4gICAgICogQHBhcmFtIHByZWZpeCBUaGUgcGF0aCB0byB0aGUgbWFpbiBKYXZhU2NyaXB0IGZpbGUncyBkaXJlY3RvcnkuXG4gICAgICovXG4gICAgZmFjdG9yeUNvbmZpZy5sb2NhdGVGaWxlID0gKHBhdGgsIHByZWZpeCkgPT4ge1xuICAgICAgaWYgKHBhdGguZW5kc1dpdGgoJy53b3JrZXIuanMnKSkge1xuICAgICAgICAvLyBFc2NhcGUgJ1xcbicgYmVjYXVzZSBCbG9iIHdpbGwgdHVybiBpdCBpbnRvIGEgbmV3bGluZS5cbiAgICAgICAgLy8gVGhlcmUgc2hvdWxkIGJlIGEgc2V0dGluZyBmb3IgdGhpcywgYnV0ICdlbmRpbmdzOiBcIm5hdGl2ZVwiJyBkb2VzXG4gICAgICAgIC8vIG5vdCBzZWVtIHRvIHdvcmsuXG4gICAgICAgIGNvbnN0IHJlc3BvbnNlID0gKHdhc21Xb3JrZXJDb250ZW50cyBhcyBzdHJpbmcpLnJlcGxhY2UoL1xcbi9nLCAnXFxcXG4nKTtcbiAgICAgICAgY29uc3QgYmxvYiA9IG5ldyBCbG9iKFtyZXNwb25zZV0sIHt0eXBlOiAnYXBwbGljYXRpb24vamF2YXNjcmlwdCd9KTtcbiAgICAgICAgcmV0dXJuIFVSTC5jcmVhdGVPYmplY3RVUkwoYmxvYik7XG4gICAgICB9XG5cbiAgICAgIGlmIChwYXRoLmVuZHNXaXRoKCcud2FzbScpKSB7XG4gICAgICAgIHJldHVybiBnZXRQYXRoVG9XYXNtQmluYXJ5KFxuICAgICAgICAgICAgc2ltZFN1cHBvcnRlZCBhcyBib29sZWFuLCB0aHJlYWRzU3VwcG9ydGVkIGFzIGJvb2xlYW4sXG4gICAgICAgICAgICB3YXNtUGF0aFByZWZpeCAhPSBudWxsID8gd2FzbVBhdGhQcmVmaXggOiBwcmVmaXgpO1xuICAgICAgfVxuICAgICAgcmV0dXJuIHByZWZpeCArIHBhdGg7XG4gICAgfTtcblxuICAgIC8vIFVzZSB0aGUgaW5zdGFudGlhdGVXYXNtIG92ZXJyaWRlIHdoZW4gc3lzdGVtIGZldGNoIGlzIG5vdCBhdmFpbGFibGUuXG4gICAgLy8gUmVmZXJlbmNlOlxuICAgIC8vIGh0dHBzOi8vZ2l0aHViLmNvbS9lbXNjcmlwdGVuLWNvcmUvZW1zY3JpcHRlbi9ibG9iLzJiY2EwODNjYmJkNWE0MTMzZGI2MWZiZDc0ZDA0ZjdmZWVjZmE5MDcvdGVzdHMvbWFudWFsX3dhc21faW5zdGFudGlhdGUuaHRtbCNMMTcwXG4gICAgaWYgKGN1c3RvbUZldGNoKSB7XG4gICAgICBmYWN0b3J5Q29uZmlnLmluc3RhbnRpYXRlV2FzbSA9XG4gICAgICAgICAgY3JlYXRlSW5zdGFudGlhdGVXYXNtRnVuYyhnZXRQYXRoVG9XYXNtQmluYXJ5KFxuICAgICAgICAgICAgICBzaW1kU3VwcG9ydGVkIGFzIGJvb2xlYW4sIHRocmVhZHNTdXBwb3J0ZWQgYXMgYm9vbGVhbixcbiAgICAgICAgICAgICAgd2FzbVBhdGhQcmVmaXggIT0gbnVsbCA/IHdhc21QYXRoUHJlZml4IDogJycpKTtcbiAgICB9XG5cbiAgICBsZXQgaW5pdGlhbGl6ZWQgPSBmYWxzZTtcbiAgICBmYWN0b3J5Q29uZmlnLm9uQWJvcnQgPSAoKSA9PiB7XG4gICAgICBpZiAoaW5pdGlhbGl6ZWQpIHtcbiAgICAgICAgLy8gRW1zY3JpcHRlbiBhbHJlYWR5IGNhbGxlZCBjb25zb2xlLndhcm4gc28gbm8gbmVlZCB0byBkb3VibGUgbG9nLlxuICAgICAgICByZXR1cm47XG4gICAgICB9XG4gICAgICBpZiAoaW5pdEFib3J0ZWQpIHtcbiAgICAgICAgLy8gRW1zY3JpcHRlbiBjYWxscyBgb25BYm9ydGAgdHdpY2UsIHJlc3VsdGluZyBpbiBkb3VibGUgZXJyb3JcbiAgICAgICAgLy8gbWVzc2FnZXMuXG4gICAgICAgIHJldHVybjtcbiAgICAgIH1cbiAgICAgIGluaXRBYm9ydGVkID0gdHJ1ZTtcbiAgICAgIGNvbnN0IHJlamVjdE1zZyA9XG4gICAgICAgICAgJ01ha2Ugc3VyZSB0aGUgc2VydmVyIGNhbiBzZXJ2ZSB0aGUgYC53YXNtYCBmaWxlIHJlbGF0aXZlIHRvIHRoZSAnICtcbiAgICAgICAgICAnYnVuZGxlZCBqcyBmaWxlLiBGb3IgbW9yZSBkZXRhaWxzIHNlZSBodHRwczovL2dpdGh1Yi5jb20vdGVuc29yZmxvdy90ZmpzL2Jsb2IvbWFzdGVyL3RmanMtYmFja2VuZC13YXNtL1JFQURNRS5tZCN1c2luZy1idW5kbGVycyc7XG4gICAgICByZWplY3Qoe21lc3NhZ2U6IHJlamVjdE1zZ30pO1xuICAgIH07XG5cbiAgICBsZXQgd2FzbTogUHJvbWlzZTxCYWNrZW5kV2FzbU1vZHVsZT47XG4gICAgLy8gSWYgYHdhc21QYXRoYCBoYXMgYmVlbiBkZWZpbmVkIHdlIG11c3QgaW5pdGlhbGl6ZSB0aGUgdmFuaWxsYSBtb2R1bGUuXG4gICAgaWYgKHRocmVhZHNTdXBwb3J0ZWQgJiYgc2ltZFN1cHBvcnRlZCAmJiB3YXNtUGF0aCA9PSBudWxsKSB7XG4gICAgICBmYWN0b3J5Q29uZmlnLm1haW5TY3JpcHRVcmxPckJsb2IgPSBuZXcgQmxvYihcbiAgICAgICAgICBbYHZhciBXYXNtQmFja2VuZE1vZHVsZVRocmVhZGVkU2ltZCA9IGAgK1xuICAgICAgICAgICB3YXNtRmFjdG9yeVRocmVhZGVkU2ltZC50b1N0cmluZygpXSxcbiAgICAgICAgICB7dHlwZTogJ3RleHQvamF2YXNjcmlwdCd9KTtcbiAgICAgIHdhc20gPSB3YXNtRmFjdG9yeVRocmVhZGVkU2ltZChmYWN0b3J5Q29uZmlnKTtcbiAgICB9IGVsc2Uge1xuICAgICAgLy8gVGhlIHdhc21GYWN0b3J5IHdvcmtzIGZvciBib3RoIHZhbmlsbGEgYW5kIFNJTUQgYmluYXJpZXMuXG4gICAgICB3YXNtID0gd2FzbUZhY3RvcnkoZmFjdG9yeUNvbmZpZyk7XG4gICAgfVxuXG4gICAgLy8gVGhlIGB3YXNtYCBwcm9taXNlIHdpbGwgcmVzb2x2ZSB0byB0aGUgV0FTTSBtb2R1bGUgY3JlYXRlZCBieVxuICAgIC8vIHRoZSBmYWN0b3J5LCBidXQgaXQgbWlnaHQgaGF2ZSBoYWQgZXJyb3JzIGR1cmluZyBjcmVhdGlvbi4gTW9zdFxuICAgIC8vIGVycm9ycyBhcmUgY2F1Z2h0IGJ5IHRoZSBvbkFib3J0IGNhbGxiYWNrIGRlZmluZWQgYWJvdmUuXG4gICAgLy8gSG93ZXZlciwgc29tZSBlcnJvcnMsIHN1Y2ggYXMgdGhvc2Ugb2NjdXJyaW5nIGZyb20gYVxuICAgIC8vIGZhaWxlZCBmZXRjaCwgcmVzdWx0IGluIHRoaXMgcHJvbWlzZSBiZWluZyByZWplY3RlZC4gVGhlc2UgYXJlXG4gICAgLy8gY2F1Z2h0IGFuZCByZS1yZWplY3RlZCBiZWxvdy5cbiAgICB3YXNtLnRoZW4oKG1vZHVsZSkgPT4ge1xuICAgICAgaW5pdGlhbGl6ZWQgPSB0cnVlO1xuICAgICAgaW5pdEFib3J0ZWQgPSBmYWxzZTtcblxuICAgICAgY29uc3Qgdm9pZFJldHVyblR5cGU6IHN0cmluZyA9IG51bGw7XG4gICAgICAvLyBVc2luZyB0aGUgdGZqcyBuYW1lc3BhY2UgdG8gYXZvaWQgY29uZmxpY3Qgd2l0aCBlbXNjcmlwdGVuJ3MgQVBJLlxuICAgICAgbW9kdWxlLnRmanMgPSB7XG4gICAgICAgIGluaXQ6IG1vZHVsZS5jd3JhcCgnaW5pdCcsIG51bGwsIFtdKSxcbiAgICAgICAgaW5pdFdpdGhUaHJlYWRzQ291bnQ6XG4gICAgICAgICAgICBtb2R1bGUuY3dyYXAoJ2luaXRfd2l0aF90aHJlYWRzX2NvdW50JywgbnVsbCwgWydudW1iZXInXSksXG4gICAgICAgIGdldFRocmVhZHNDb3VudDogbW9kdWxlLmN3cmFwKCdnZXRfdGhyZWFkc19jb3VudCcsICdudW1iZXInLCBbXSksXG4gICAgICAgIHJlZ2lzdGVyVGVuc29yOiBtb2R1bGUuY3dyYXAoXG4gICAgICAgICAgICAncmVnaXN0ZXJfdGVuc29yJywgbnVsbCxcbiAgICAgICAgICAgIFtcbiAgICAgICAgICAgICAgJ251bWJlcicsICAvLyBpZFxuICAgICAgICAgICAgICAnbnVtYmVyJywgIC8vIHNpemVcbiAgICAgICAgICAgICAgJ251bWJlcicsICAvLyBtZW1vcnlPZmZzZXRcbiAgICAgICAgICAgIF0pLFxuICAgICAgICBkaXNwb3NlRGF0YTogbW9kdWxlLmN3cmFwKCdkaXNwb3NlX2RhdGEnLCB2b2lkUmV0dXJuVHlwZSwgWydudW1iZXInXSksXG4gICAgICAgIGRpc3Bvc2U6IG1vZHVsZS5jd3JhcCgnZGlzcG9zZScsIHZvaWRSZXR1cm5UeXBlLCBbXSksXG4gICAgICB9O1xuXG4gICAgICByZXNvbHZlKHt3YXNtOiBtb2R1bGV9KTtcbiAgICB9KS5jYXRjaChyZWplY3QpO1xuICB9KTtcbn1cblxuZnVuY3Rpb24gdHlwZWRBcnJheUZyb21CdWZmZXIoXG4gICAgYnVmZmVyOiBBcnJheUJ1ZmZlciwgZHR5cGU6IERhdGFUeXBlKTogYmFja2VuZF91dGlsLlR5cGVkQXJyYXkge1xuICBzd2l0Y2ggKGR0eXBlKSB7XG4gICAgY2FzZSAnZmxvYXQzMic6XG4gICAgICByZXR1cm4gbmV3IEZsb2F0MzJBcnJheShidWZmZXIpO1xuICAgIGNhc2UgJ2ludDMyJzpcbiAgICAgIHJldHVybiBuZXcgSW50MzJBcnJheShidWZmZXIpO1xuICAgIGNhc2UgJ2Jvb2wnOlxuICAgICAgcmV0dXJuIG5ldyBVaW50OEFycmF5KGJ1ZmZlcik7XG4gICAgZGVmYXVsdDpcbiAgICAgIHRocm93IG5ldyBFcnJvcihgVW5rbm93biBkdHlwZSAke2R0eXBlfWApO1xuICB9XG59XG5cbmNvbnN0IHdhc21CaW5hcnlOYW1lcyA9IFtcbiAgJ3RmanMtYmFja2VuZC13YXNtLndhc20nLCAndGZqcy1iYWNrZW5kLXdhc20tc2ltZC53YXNtJyxcbiAgJ3RmanMtYmFja2VuZC13YXNtLXRocmVhZGVkLXNpbWQud2FzbSdcbl0gYXMgY29uc3QgO1xudHlwZSBXYXNtQmluYXJ5TmFtZSA9IHR5cGVvZiB3YXNtQmluYXJ5TmFtZXNbbnVtYmVyXTtcblxubGV0IHdhc21QYXRoOiBzdHJpbmcgPSBudWxsO1xubGV0IHdhc21QYXRoUHJlZml4OiBzdHJpbmcgPSBudWxsO1xubGV0IHdhc21GaWxlTWFwOiB7W2tleSBpbiBXYXNtQmluYXJ5TmFtZV0/OiBzdHJpbmd9ID0ge307XG5sZXQgaW5pdEFib3J0ZWQgPSBmYWxzZTtcbmxldCBjdXN0b21GZXRjaCA9IGZhbHNlO1xuXG4vKipcbiAqIEBkZXByZWNhdGVkIFVzZSBgc2V0V2FzbVBhdGhzYCBpbnN0ZWFkLlxuICogU2V0cyB0aGUgcGF0aCB0byB0aGUgYC53YXNtYCBmaWxlIHdoaWNoIHdpbGwgYmUgZmV0Y2hlZCB3aGVuIHRoZSB3YXNtXG4gKiBiYWNrZW5kIGlzIGluaXRpYWxpemVkLiBTZWVcbiAqIGh0dHBzOi8vZ2l0aHViLmNvbS90ZW5zb3JmbG93L3RmanMvYmxvYi9tYXN0ZXIvdGZqcy1iYWNrZW5kLXdhc20vUkVBRE1FLm1kI3VzaW5nLWJ1bmRsZXJzXG4gKiBmb3IgbW9yZSBkZXRhaWxzLlxuICogQHBhcmFtIHBhdGggd2FzbSBmaWxlIHBhdGggb3IgdXJsXG4gKiBAcGFyYW0gdXNlUGxhdGZvcm1GZXRjaCBvcHRpb25hbCBib29sZWFuIHRvIHVzZSBwbGF0Zm9ybSBmZXRjaCB0byBkb3dubG9hZFxuICogICAgIHRoZSB3YXNtIGZpbGUsIGRlZmF1bHQgdG8gZmFsc2UuXG4gKlxuICogQGRvYyB7aGVhZGluZzogJ0Vudmlyb25tZW50JywgbmFtZXNwYWNlOiAnd2FzbSd9XG4gKi9cbmV4cG9ydCBmdW5jdGlvbiBzZXRXYXNtUGF0aChwYXRoOiBzdHJpbmcsIHVzZVBsYXRmb3JtRmV0Y2ggPSBmYWxzZSk6IHZvaWQge1xuICBkZXByZWNhdGlvbldhcm4oXG4gICAgICAnc2V0V2FzbVBhdGggaGFzIGJlZW4gZGVwcmVjYXRlZCBpbiBmYXZvciBvZiBzZXRXYXNtUGF0aHMgYW5kJyArXG4gICAgICAnIHdpbGwgYmUgcmVtb3ZlZCBpbiBhIGZ1dHVyZSByZWxlYXNlLicpO1xuICBpZiAoaW5pdEFib3J0ZWQpIHtcbiAgICB0aHJvdyBuZXcgRXJyb3IoXG4gICAgICAgICdUaGUgV0FTTSBiYWNrZW5kIHdhcyBhbHJlYWR5IGluaXRpYWxpemVkLiBNYWtlIHN1cmUgeW91IGNhbGwgJyArXG4gICAgICAgICdgc2V0V2FzbVBhdGgoKWAgYmVmb3JlIHlvdSBjYWxsIGB0Zi5zZXRCYWNrZW5kKClgIG9yIGB0Zi5yZWFkeSgpYCcpO1xuICB9XG4gIHdhc21QYXRoID0gcGF0aDtcbiAgY3VzdG9tRmV0Y2ggPSB1c2VQbGF0Zm9ybUZldGNoO1xufVxuXG4vKipcbiAqIENvbmZpZ3VyZXMgdGhlIGxvY2F0aW9ucyBvZiB0aGUgV0FTTSBiaW5hcmllcy5cbiAqXG4gKiBgYGBqc1xuICogc2V0V2FzbVBhdGhzKHtcbiAqICAndGZqcy1iYWNrZW5kLXdhc20ud2FzbSc6ICdyZW5hbWVkLndhc20nLFxuICogICd0ZmpzLWJhY2tlbmQtd2FzbS1zaW1kLndhc20nOiAncmVuYW1lZC1zaW1kLndhc20nLFxuICogICd0ZmpzLWJhY2tlbmQtd2FzbS10aHJlYWRlZC1zaW1kLndhc20nOiAncmVuYW1lZC10aHJlYWRlZC1zaW1kLndhc20nXG4gKiB9KTtcbiAqIHRmLnNldEJhY2tlbmQoJ3dhc20nKTtcbiAqIGBgYFxuICpcbiAqIEBwYXJhbSBwcmVmaXhPckZpbGVNYXAgVGhpcyBjYW4gYmUgZWl0aGVyIGEgc3RyaW5nIG9yIG9iamVjdDpcbiAqICAtIChzdHJpbmcpIFRoZSBwYXRoIHRvIHRoZSBkaXJlY3Rvcnkgd2hlcmUgdGhlIFdBU00gYmluYXJpZXMgYXJlIGxvY2F0ZWQuXG4gKiAgICAgTm90ZSB0aGF0IHRoaXMgcHJlZml4IHdpbGwgYmUgdXNlZCB0byBsb2FkIGVhY2ggYmluYXJ5ICh2YW5pbGxhLFxuICogICAgIFNJTUQtZW5hYmxlZCwgdGhyZWFkaW5nLWVuYWJsZWQsIGV0Yy4pLlxuICogIC0gKG9iamVjdCkgTWFwcGluZyBmcm9tIG5hbWVzIG9mIFdBU00gYmluYXJpZXMgdG8gY3VzdG9tXG4gKiAgICAgZnVsbCBwYXRocyBzcGVjaWZ5aW5nIHRoZSBsb2NhdGlvbnMgb2YgdGhvc2UgYmluYXJpZXMuIFRoaXMgaXMgdXNlZnVsIGlmXG4gKiAgICAgeW91ciBXQVNNIGJpbmFyaWVzIGFyZSBub3QgYWxsIGxvY2F0ZWQgaW4gdGhlIHNhbWUgZGlyZWN0b3J5LCBvciBpZiB5b3VyXG4gKiAgICAgV0FTTSBiaW5hcmllcyBoYXZlIGJlZW4gcmVuYW1lZC5cbiAqIEBwYXJhbSB1c2VQbGF0Zm9ybUZldGNoIG9wdGlvbmFsIGJvb2xlYW4gdG8gdXNlIHBsYXRmb3JtIGZldGNoIHRvIGRvd25sb2FkXG4gKiAgICAgdGhlIHdhc20gZmlsZSwgZGVmYXVsdCB0byBmYWxzZS5cbiAqXG4gKiBAZG9jIHtoZWFkaW5nOiAnRW52aXJvbm1lbnQnLCBuYW1lc3BhY2U6ICd3YXNtJ31cbiAqL1xuZXhwb3J0IGZ1bmN0aW9uIHNldFdhc21QYXRocyhcbiAgICBwcmVmaXhPckZpbGVNYXA6IHN0cmluZ3x7W2tleSBpbiBXYXNtQmluYXJ5TmFtZV0/OiBzdHJpbmd9LFxuICAgIHVzZVBsYXRmb3JtRmV0Y2ggPSBmYWxzZSk6IHZvaWQge1xuICBpZiAoaW5pdEFib3J0ZWQpIHtcbiAgICB0aHJvdyBuZXcgRXJyb3IoXG4gICAgICAgICdUaGUgV0FTTSBiYWNrZW5kIHdhcyBhbHJlYWR5IGluaXRpYWxpemVkLiBNYWtlIHN1cmUgeW91IGNhbGwgJyArXG4gICAgICAgICdgc2V0V2FzbVBhdGhzKClgIGJlZm9yZSB5b3UgY2FsbCBgdGYuc2V0QmFja2VuZCgpYCBvciAnICtcbiAgICAgICAgJ2B0Zi5yZWFkeSgpYCcpO1xuICB9XG5cbiAgaWYgKHR5cGVvZiBwcmVmaXhPckZpbGVNYXAgPT09ICdzdHJpbmcnKSB7XG4gICAgd2FzbVBhdGhQcmVmaXggPSBwcmVmaXhPckZpbGVNYXA7XG4gIH0gZWxzZSB7XG4gICAgd2FzbUZpbGVNYXAgPSBwcmVmaXhPckZpbGVNYXA7XG4gICAgY29uc3QgbWlzc2luZ1BhdGhzID1cbiAgICAgICAgd2FzbUJpbmFyeU5hbWVzLmZpbHRlcihuYW1lID0+IHdhc21GaWxlTWFwW25hbWVdID09IG51bGwpO1xuICAgIGlmIChtaXNzaW5nUGF0aHMubGVuZ3RoID4gMCkge1xuICAgICAgdGhyb3cgbmV3IEVycm9yKFxuICAgICAgICAgIGBUaGVyZSB3ZXJlIG5vIGVudHJpZXMgZm91bmQgZm9yIHRoZSBmb2xsb3dpbmcgYmluYXJpZXM6IGAgK1xuICAgICAgICAgIGAke21pc3NpbmdQYXRocy5qb2luKCcsJyl9LiBQbGVhc2UgZWl0aGVyIGNhbGwgc2V0V2FzbVBhdGhzIHdpdGggYSBgICtcbiAgICAgICAgICBgbWFwIHByb3ZpZGluZyBhIHBhdGggZm9yIGVhY2ggYmluYXJ5LCBvciB3aXRoIGEgc3RyaW5nIGluZGljYXRpbmcgYCArXG4gICAgICAgICAgYHRoZSBkaXJlY3Rvcnkgd2hlcmUgYWxsIHRoZSBiaW5hcmllcyBjYW4gYmUgZm91bmQuYCk7XG4gICAgfVxuICB9XG5cbiAgY3VzdG9tRmV0Y2ggPSB1c2VQbGF0Zm9ybUZldGNoO1xufVxuXG4vKiogVXNlZCBpbiB1bml0IHRlc3RzLiAqL1xuZXhwb3J0IGZ1bmN0aW9uIHJlc2V0V2FzbVBhdGgoKTogdm9pZCB7XG4gIHdhc21QYXRoID0gbnVsbDtcbiAgd2FzbVBhdGhQcmVmaXggPSBudWxsO1xuICB3YXNtRmlsZU1hcCA9IHt9O1xuICBjdXN0b21GZXRjaCA9IGZhbHNlO1xuICBpbml0QWJvcnRlZCA9IGZhbHNlO1xufVxuXG5sZXQgdGhyZWFkc0NvdW50ID0gLTE7XG5sZXQgYWN0dWFsVGhyZWFkc0NvdW50ID0gLTE7XG5cbi8qKlxuICogU2V0cyB0aGUgbnVtYmVyIG9mIHRocmVhZHMgdGhhdCB3aWxsIGJlIHVzZWQgYnkgWE5OUEFDSyB0byBjcmVhdGVcbiAqIHRocmVhZHBvb2wgKGRlZmF1bHQgdG8gdGhlIG51bWJlciBvZiBsb2dpY2FsIENQVSBjb3JlcykuXG4gKlxuICogVGhpcyBtdXN0IGJlIGNhbGxlZCBiZWZvcmUgY2FsbGluZyBgdGYuc2V0QmFja2VuZCgnd2FzbScpYC5cbiAqL1xuZXhwb3J0IGZ1bmN0aW9uIHNldFRocmVhZHNDb3VudChudW1UaHJlYWRzOiBudW1iZXIpIHtcbiAgdGhyZWFkc0NvdW50ID0gbnVtVGhyZWFkcztcbn1cblxuLyoqXG4gKiBHZXRzIHRoZSBhY3R1YWwgdGhyZWFkcyBjb3VudCB0aGF0IGlzIHVzZWQgYnkgWE5OUEFDSy5cbiAqXG4gKiBJdCBpcyBzZXQgYWZ0ZXIgdGhlIGJhY2tlbmQgaXMgaW50aWFsaXplZC5cbiAqL1xuZXhwb3J0IGZ1bmN0aW9uIGdldFRocmVhZHNDb3VudCgpOiBudW1iZXIge1xuICBpZiAoYWN0dWFsVGhyZWFkc0NvdW50ID09PSAtMSkge1xuICAgIHRocm93IG5ldyBFcnJvcihgV0FTTSBiYWNrZW5kIG5vdCBpbml0aWFsaXplZC5gKTtcbiAgfVxuICByZXR1cm4gYWN0dWFsVGhyZWFkc0NvdW50O1xufVxuIl19