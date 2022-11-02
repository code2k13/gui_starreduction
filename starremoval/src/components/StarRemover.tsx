import { IonButton, IonGrid, IonRow, IonCol, IonLabel, IonContent, IonItem, IonRadioGroup, IonRadio, IonList, IonIcon, UseIonLoadingResult, IonLoading, IonBackdrop, IonModal } from '@ionic/react';
import React, { useEffect } from 'react';
import * as tf from '@tensorflow/tfjs';
import { Tensor3D } from '@tensorflow/tfjs';
//import '@tensorflow/tfjs-backend-wasm'

tf.setBackend('webgl')

class StarRemover extends React.Component<any, any> {

    private model: any = null;
    private refimg: any = null;
    private refcanvas: any = null;
    private refimg3: any = null;

    constructor(props: any) {
        super(props);
        this.state = { showLoading: true, inputMode: 'random', sourceImage: "sample.jpg" }
        this.handleChange = this.handleChange.bind(this);
        this.convert = this.convert.bind(this);
        this.componentDidMount = this.componentDidMount.bind(this);
        this.handleRadioChange = this.handleRadioChange.bind(this);
        this.refimg = React.createRef();
        this.refcanvas = React.createRef();
        this.refimg3 = React.createRef();
    }

    private async convert(model: any) {
        let img: any = document.getElementById('img');
        img = tf.browser.fromPixels(img, 3)
        let channels = []
        for (let i = 0; i < 3; i++) {
            let channel = tf.slice(img, [0, 0, i], [512, 512, 1])
            channel = tf.reshape(channel, [1, 512, 512, 1]);
            channel = tf.div(channel, tf.scalar(1024));
            //@ts-ignore
            let a = await window.tfmodel.predict([channel]) as Tensor3D;
            a = tf.reshape(a, [512, 512, 1])
            channels[i] = tf.mul(a, tf.scalar(7.80))
        }

        let tf_image = tf.concat([channels[0], channels[1], channels[2]], 2);
        tf_image = tf.reshape(tf_image, [512, 512, 3])
        let canvas: any = document.getElementById("output")
        await tf.browser.toPixels(tf_image, canvas).then(() => {
            document.getElementById("img3")?.setAttribute("src", canvas.toDataURL("image/jpeg"))
            //this.setState({ showLoading: false }, () => this.forceUpdate())

        })
    }

    handleChange(event: any) {
        this.setState({ sourceImage: URL.createObjectURL(event.target.files[0]) });
    }

    handleRadioChange(event: any) {
        console.log(event.target.value)
        this.setState({ inputMode: event.target.value })
        if (event.target.value == 'random') {
            this.setState({ sourceImage: "sample.jpg" })
        }
        this.forceUpdate()
        event.preventDefault();
    }

    async handleBeginConvert(event: any) {
        this.setState({ showBusy: true }, async () => {
            await this.convert(this.model)
            this.setState({ enableDownload: true, showBusy: false })
        });
    }

    async handleDownloadClick(event: any) {
        const a = document.createElement("a");
        //@ts-ignore
        a.href = document.getElementById('output').toDataURL()
        a.download = "starless.png";
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
    }

    componentDidMount() {
        //@ts-ignore
        if (!window.tfmodel) {
            //@ts-ignore
            window.tfmodel = tf.loadLayersModel('model_js/model.json', {
                onProgress: (n) => {
                    console.log("Model downloaded : " + (n * 100).toString() + " %")
                }
            }
            ).then((model) => {
                console.log("model loading completed !")
                //@ts-ignore
                window.tfmodel = model;
                this.setState({ showLoading: false }, () => {
                    this.forceUpdate()
                });

            })
        } else {
            //@ts-ignore
            console.log("model has already been loaded !")
            //@ts-ignore
            this.model = window.tfmodel;
            window.setTimeout(() => {
                this.setState({ showLoading: false });
                this.forceUpdate()
            }, 100);
        }
    }

    render() {
        return (
            <IonContent>
                <IonGrid>
                    <IonRow>
                        <IonCol> <IonLoading
                            isOpen={this.state.showLoading}
                            duration={2000}
                            message='Loading model. This can take a couple of minutes !'
                        />

                            <IonLoading
                                isOpen={this.state.showBusy}
                                duration={5000}
                                message='Processing ....'
                            />

                            <IonList>
                                <IonRadioGroup value={this.state.inputMode} onIonChange={this.handleRadioChange}>
                                    <IonItem>
                                        <IonLabel>Upload image</IonLabel>
                                        <IonRadio slot="end" value="upload"></IonRadio>
                                    </IonItem>

                                    <IonItem>
                                        <IonLabel>Use sample image</IonLabel>
                                        <IonRadio slot="end" value="random"></IonRadio>
                                    </IonItem>
                                </IonRadioGroup>
                            </IonList></IonCol>
                        <IonCol>
                            {this.state.inputMode == 'upload' && <input type="file" accept=".jpg, .png, .jpeg, .gif, .bmp, .tif, .tiff|image/*" onChange={this.handleChange}></input>
                            } <IonButton size="small" onClick={this.handleBeginConvert.bind(this)}> Remove Stars </IonButton>
                        </IonCol>

                    </IonRow>
                    <IonRow style={{ height: "0px" }}>
                        <IonCol><img id="img" ref={this.refimg} src={this.state.sourceImage} style={{ width: "512px", height: "512px", maxWidth: "512px", visibility: "hidden" }}></img></IonCol>
                        <IonCol><canvas ref={this.refcanvas} id="output" style={{ width: "512px", height: "512px", maxWidth: "512px", visibility: "hidden" }}></canvas></IonCol>
                    </IonRow>
                    <IonRow>
                        <IonCol> <IonLabel color="dark">Original Image</IonLabel></IonCol>
                        <IonCol>   <IonLabel color="dark">Starless Image  </IonLabel></IonCol>

                    </IonRow>
                    <IonRow>
                        <IonCol><img id="img2" src={this.state.sourceImage} style={{ width: "100%" }}></img></IonCol>
                        <IonCol><img id="img3" ref={this.refimg3} style={{ width: "100%" }}></img>
                            <IonButton fill="outline" disabled={!this.state.enableDownload} size="small" onClick={this.handleDownloadClick.bind(this)}>Download</IonButton></IonCol>
                    </IonRow>
                </IonGrid>
            </IonContent>
        );
    }
}

export default StarRemover;