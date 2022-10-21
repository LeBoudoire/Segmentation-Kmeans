#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <math.h>   
#include <iostream>
#include <limits>
#include <cmath>
#include <time.h>
#include <stdlib.h>
#include <array>

using namespace cv;
using namespace std;

// Déclarations
Mat image, samples, bestLabels, centers;
int k;
Mat myCenters, myLabels;

// Afficher l'usage
void printHelp(const string& progName)
{
    cout << "Usage:\n\t " << progName << " <image_file> <K_num_of_clusters> [<image_ground_truth>]" << endl;
}

// Transformer une image à une matrice avec les pixels sur les lignes et leurs valeurs RGB sur les colonnes
// @param	Mat			l'image
// @return	Mat         l'image linéarisée
Mat linearizeImage(Mat image){

    Mat samples(image.rows * image.cols, 3, CV_32F);
    for( int y = 0; y < image.rows; y++ )
    for( int x = 0; x < image.cols; x++ )
        for( int z = 0; z < 3; z++)
            samples.at<float>(y + x*image.rows, z) = image.at<Vec3b>(y,x)[z];
    return samples;

}

// Récupère le résultat de kmeans et produit l'image segmentée pour la visualisation
// @param	Mat			l'image initiale
// @param	Mat			l'ensemble des centres
// @param	Mat			les labels des pixels indiquant les régions auxquelles ils appartient
// @param	int			le nombre de régions
// @return	Mat         l'image segmentée
Mat buildSegmentedImage(Mat image, Mat centers, Mat bestLabels, int k){

    Mat new_image(image.size(),image.type());
    for( int y = 0; y < image.rows; y++ )
    for( int x = 0; x < image.cols; x++ )
    {
        int cluster_idx = bestLabels.at<int>(y + x*image.rows,0);
        if (k>2) { // Cas d'une segmentation binaire (choix entre blanc ou noir)
            new_image.at<Vec3b>(y,x)[0] = centers.at<float>(cluster_idx, 0);
            new_image.at<Vec3b>(y,x)[1] = centers.at<float>(cluster_idx, 1);
            new_image.at<Vec3b>(y,x)[2] = centers.at<float>(cluster_idx, 2);
        } else { // Cas général (couleur du centre)
            if (cluster_idx == 0) {
                new_image.at<Vec3b>(y,x)[0] = 255;
                new_image.at<Vec3b>(y,x)[1] = 255;
                new_image.at<Vec3b>(y,x)[2] = 255;
            } else {
                new_image.at<Vec3b>(y,x)[0] = 0;
                new_image.at<Vec3b>(y,x)[1] = 0;
                new_image.at<Vec3b>(y,x)[2] = 0;
            }        
        }
    }
    return new_image;

}

// Distance entre 2 pixels
// @param	Vec3b	un point
// @param	Vec3b	un autre point
// @return	double	distance entre ces deux points
float dist(Vec3b p1, Vec3b p2){
    float diffB = p1[0] - p2[0];
    float diffG = p1[1] - p2[1];
    float diffR = p1[2] - p2[2];

    float dist = sqrt(pow(diffB, 2) + pow(diffG, 2) + pow(diffR, 2));
    return dist;
}

// Initialisation des données pour l'algorithme de kmeans
void initialize() {

    float tailleChunk = ceil(255/k);
    float diff = 255%k;

    // Decoupage les niveaux de couleurs en k parties
    for (int i = 0; i < k; i++) 
    {
        float sum = 0;

        if (i == k-1) {
            tailleChunk = tailleChunk - k + diff;  
        }

        for ( int j = i*tailleChunk; j < (i+1)*tailleChunk; j++ )
        {
            sum += j;
        }

        // Définir le centre
        myCenters.at<float>(i, 0) = sum/tailleChunk;
        myCenters.at<float>(i, 1) = sum/tailleChunk;
        myCenters.at<float>(i, 2) = sum/tailleChunk;

    }

}

// Mise à jour des clusters
void updateClusters() {

    // Vider les clusters
    for( int y = 0; y < image.rows; y++ )
    for( int x = 0; x < image.cols; x++ )
    {
        myLabels.at<int>(y + x*image.rows,0) = -1;
    }

    // Parcourir tous les pixels de l'image
    for (int y = 0 ; y < image.rows; y++) {
        for (int x = 0; x < image.cols; x++) {

            float distMin = INFINITY;
            int nearestClusterIndex = 0;
            Vec3b pixel = image.at<Vec3b>(y,x);

            // Chercher le centre le plus proche
            for (int i = 0; i < k; i++) {
                Vec3b clusterCenter(myCenters.at<float>(i, 0), myCenters.at<float>(i, 1), myCenters.at<float>(i, 2));
                float distance = dist(pixel, clusterCenter);

                if (distance < distMin) {
                    distMin = distance;
                    nearestClusterIndex = i;
                }
            }

            // Mise à jour du label
            myLabels.at<int>(y + x*image.rows,0) = nearestClusterIndex;
        }
    }

}

// Mise à jour des centres
// Renvoie une moyenne sur les modifications des centres
float updateCenters() {

    float change;
    Mat newCenters = Mat::zeros(k, 3, CV_32F);
    vector<float> counter(k);

    // Calucler les nouveaux centres
    for (int y = 0 ; y < image.rows; y++) {
        for (int x = 0; x < image.cols; x++) {
            int k = myLabels.at<int>(y + x*image.rows,0);
            Vec3b pixel = image.at<Vec3b>(y, x);
            newCenters.at<float>(k, 0) += pixel.val[0];
            newCenters.at<float>(k, 1) += pixel.val[1];
            newCenters.at<float>(k, 2) += pixel.val[2];
            counter[k]++;
        }
    }

    for (int i = 0; i < k; i++) {
        // Diviser par le nombre de pixels de la région
        newCenters.at<float>(i, 0) /= counter[i];
        newCenters.at<float>(i, 1) /= counter[i];
        newCenters.at<float>(i, 2) /= counter[i];

        // Calculer la distance par rapport à l'ancien centre
        Vec3b newCenter(newCenters.at<float>(i, 0), newCenters.at<float>(i, 1), newCenters.at<float>(i, 2));
        Vec3b oldCenter(myCenters.at<float>(i, 0), myCenters.at<float>(i, 1), myCenters.at<float>(i, 2));
        change += dist(newCenter, oldCenter);
    }

    // Mise à jour des centres
    newCenters.copyTo(myCenters);

    // Renvoyer la moyenne sur l'écart entre les nouveaux et anciens centres
    change /= k;
    return change;
}

// Evaluer la qualité d'une segmentation
// @param	Mat					image segmentée
// @param	Mat					image de référence
// @return	array<float, 3>		[precision, sensibilité, similarité]
array<float, 3> evalSegmentation(Mat segmented, Mat reference) {

    array<float, 3> stat, stat_inv;
    Mat segmentedInverted, segmentedEval;

    // Régler le problème où l'image segemntée est inversée par rapport à la référence
    // On choisit la plus proche à la reference
    bitwise_not(segmented, segmentedInverted);
    double diffNotInverted = sum(abs(segmented - reference))[0];
    double diffInverted = sum(abs(segmentedInverted - reference))[0];
    if (diffNotInverted < diffInverted) {
        segmentedEval = segmented;
    } else {
        segmentedEval = segmentedInverted;
    }

    int TP = 0; int FP = 0; int FN = 0; int TN = 0;
    for(int r = 0 ; r < reference.rows; r++){
        for(int c = 0; c < reference.cols; c++){
            if(segmentedEval.at<uchar>(r, c) == 255 && reference.at<uchar>(r, c) == 255){
                TP++;
            }
            if(segmentedEval.at<uchar>(r, c) == 255 && reference.at<uchar>(r, c) == 0){
                FP++;
            }
            if(segmentedEval.at<uchar>(r, c) == 0 && reference.at<uchar>(r, c) == 255){
                FN++;
            }
            if(segmentedEval.at<uchar>(r, c) == 0 && reference.at<uchar>(r, c) == 0){
                TN++;
            }
        }
    }

    stat[0] = (float) TP / (float) (TP + FP);
    stat[1] = (float) TP / (float) (TP + FN);
    stat[2] = (float) 2*TP / (float) (2*TP + FP + FN);

	return stat;
}


// Le programme principal
int main(int argc, char** argv)
{
    if (argc != 3 && argc !=4)
    {
        cout << " Incorrect number of arguments." << endl;
        printHelp(string(argv[0]));
        return EXIT_FAILURE;
    }

    const auto imageFilename = string(argv[1]);
    const string groundTruthFilename = (argc ==4) ? string(argv[3]) : string();
    k = stoi(argv[2]);

    // Lire le fichier
    image = imread(imageFilename, CV_LOAD_IMAGE_COLOR);

    // Verifier si l'image est valide
    if(image.empty())
    {
        cout << "Could not open or find the image" << std::endl;
        return EXIT_FAILURE;
    }   

    // Lineariser l'image pour être utilisé par kmeans
    samples = linearizeImage(image);

    // Appliquer de kmeans de OpenCV
    kmeans(samples, k, bestLabels, TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 10000, 0.0001), 100, KMEANS_PP_CENTERS, centers);

    // Algorithme de kmeans
    int iterMax = 10000;
    float seuil = 0.0001;

    myCenters.create(k, 3, CV_32F);
    myLabels.create(image.rows * image.cols, 1, CV_32SC1);
	initialize();

    int iter = 0;
    float change = 1;
    while (change > seuil && iter < iterMax) {
        iter++;
        updateClusters();
        change = updateCenters();
    }

    // Recupérer les images segmentées
    Mat segmentedImageOpenCV = buildSegmentedImage(image, centers, bestLabels, k);
    Mat segmentedImageAlgo = buildSegmentedImage(image, myCenters, myLabels, k);
    //segmentedImageAlgo = Mat(image.size(),image.type(), Scalar::all(255)) - segmentedImageAlgo;

    // Enregistrer les images segmentées
    char* ImageFileName = argv[1];
    char size = strlen(ImageFileName);
    ImageFileName[size-4]='\0';
    strcat(ImageFileName, "_kmeansOpenCV.png");
    imwrite(ImageFileName, segmentedImageOpenCV);
    ImageFileName[size-4]='\0';
    strcat(ImageFileName, "_kmeansAlgo.png");
    imwrite(ImageFileName, segmentedImageAlgo);

    // Affichage des résultats
    imshow("Kmeans OpenCV", segmentedImageOpenCV );
    imshow("Kmeans Algo", segmentedImageAlgo );
    waitKey( 0 );
    destroyAllWindows();

    // Evaluer la qualité de segmentation binaire si jamais on a une image de référence
	if(!groundTruthFilename.empty() && k == 2) {

        cout << "Evaluation de la qualité de la segmentation binaire" << endl;
        // Lire l'image de référence
        Mat referenceImage = imread(groundTruthFilename, CV_LOAD_IMAGE_GRAYSCALE);

        // Convertir les images segmentées
        Mat segmentedImageOpenCV_G, segmentedImageAlgo_G;
        cvtColor(segmentedImageOpenCV, segmentedImageOpenCV_G, COLOR_BGR2GRAY);
        cvtColor(segmentedImageAlgo, segmentedImageAlgo_G, COLOR_BGR2GRAY);

        cout << "Segmentation avec kmeans de OpenCV" << endl;
        array<float, 3> statOpenCV = evalSegmentation(segmentedImageOpenCV_G, referenceImage);
        cout << "   Précision : " 		<< statOpenCV[0] << endl;
        cout << "   Sensibilité : " 	<< statOpenCV[1] << endl;
        cout << "   Similarité : " 	    << statOpenCV[2] << endl;
        cout << endl;

        cout << "Segmentation avec kmeans implémenté" << endl;
        array<float, 3> statAlgo = evalSegmentation(segmentedImageAlgo_G, referenceImage);
        cout << "   Précision : " 		<< statAlgo[0] << endl;
        cout << "   Sensibilité : " 	<< statAlgo[1] << endl;
        cout << "   Similarité : " 	    << statAlgo[2] << endl;
        cout << endl;

    }

    return EXIT_SUCCESS;
}
