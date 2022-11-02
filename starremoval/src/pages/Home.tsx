import { IonCard, IonCardContent, IonCardHeader, IonCardSubtitle, IonCardTitle, IonHeader, IonNote, IonPage, IonText, IonTitle, IonToolbar } from '@ionic/react';
import { IonContent } from '@ionic/react';
import StarRemover from '../components/StarRemover';
import './Home.css';

const Home: React.FC = () => {
  return (
    <IonPage>
      <IonHeader>
        <IonToolbar>
          <IonTitle>Star Removal</IonTitle>
        </IonToolbar>
      </IonHeader>
      <IonContent fullscreen>
        <IonHeader collapse="condense">
          <IonToolbar>
            <IonTitle size="large">Star Reduction</IonTitle>
          </IonToolbar>
        </IonHeader>
        <IonContent>



          <IonCard>          

            <IonCardContent>
              This is a browser based demonstration of my <a href='https://github.com/code2k13/starreduction'>star removal software</a> . You can upload an image from your computer
              or use a sample to test the model. For best results select a square image of upto 1024*1024px.
              All the processing happens locally within your browser.
            </IonCardContent>
          </IonCard>



          <StarRemover></StarRemover>
        </IonContent>
      </IonContent>
    </IonPage>
  );
};

export default Home;
