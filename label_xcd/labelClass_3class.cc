/*!
 * @file labelclass.cc
 * @author Zhixiang Ren
 * @date 11 Apr 2018
 * @brief Take in reconstructed xcd file, label each event's class,
 *        and output a xcd file. 
 */

#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <sstream>
#include <functional>   
#include <algorithm>    
#include <cmath>
#include <map>

#include <hawcnest/CommandLineConfigurator.h>
#include <xcdf/XCDF.h>
#include <xcdf/utility/XCDFUtility.h>
#include <hawcnest/xml/XMLReader.h>

using namespace std;



// Copy head and fields to output xcd file
void CopyComments(XCDFFile& input, XCDFFile& output)
{
    for (vector<string>::const_iterator iC = input.CommentsBegin();
        iC != input.CommentsEnd(); ++iC)
    {
        output.AddComment(*iC);
    }
}



// Parse xml string to vector
vector<double> extractVaules(string raw_str)
{
    // create vector for output
    vector<double> vec;

    // replace "[" and "]" to read values correctly
    std::replace(raw_str.begin(), raw_str.end(), '[', ' ');
    std::replace(raw_str.begin(), raw_str.end(), ']', ' ');

    // Storing the whole string into string stream
    stringstream ss;    
    ss << raw_str;
     
    // Running loop till the end of the stream
    string temp;
    double found;
    while (!ss.eof()) {
        // extracting word by word from stream 
        ss >> temp;
        // Checking the given word is a number or not
        if (stringstream(temp) >> found)
        vec.push_back(found);
        // To save from space at the end of string
        temp = "";
    }
    return vec;
}



// Get 2D arrays (weights), use "&" for things you want to change
void ParseXML_2Darray(string& xmlFileName, 
                    vector< vector<double> > & w0_mat, vector< vector<double> > & w1_mat)
{
    XMLReader r(xmlFileName, XMLReader::NONE);
    XMLBranch topB = r.GetTopBranch();

    // read variables from xml (as strings)
    string w0_str, w1_str;
    topB.GetChild("weight_0").GetData(w0_str);
    topB.GetChild("weight_1").GetData(w1_str);

    // create w0 matrix (2D vector) -------------------------------------------
    vector<double> w0_vec = extractVaules(w0_str);
    int w0_r = 40; // row of w0 (number of features)
    int w0_c = 21; // column of w0 (number of neurons in the hidden layer)

    // check if the total number of variables is correct
    unsigned int wvalues0 = w0_r * w0_c;
    if (w0_vec.size() != wvalues0) 
    { cout << "w0 has: " << w0_vec.size() << endl; log_fatal("w0 total value is wrong. Quitting !"); }

    // convert values from 1D vector to 2D array with proper dimension
    double w0_arr[w0_r][w0_c];
    for (int i=0; i<w0_r; i++){
        for (int j=0; j<w0_c; j++){
            int a = w0_c*i + j;
            w0_arr[i][j] = w0_vec[a]; }
    }

    // convert 2D array to 2D vector
    for (int n = 0; n < w0_r; ++n)
        w0_mat.push_back(vector<double>(w0_arr[n], w0_arr[n]+w0_c));

    // create w1 matrix (2D vector) -------------------------------------------
    vector<double> w1_vec = extractVaules(w1_str);
    int w1_r = 21; // row of w1 (number of neurons in the hidden layer)
    int w1_c = 3; // column of w1 (number of output classes)

    // check if the total number of variables is correct
    unsigned int wvalues1 = w1_r * w1_c;
    if (w1_vec.size() != wvalues1) 
    { cout << "w1 has: " << w1_vec.size() << endl; log_fatal("w1 total value is wrong. Quitting !"); }

    // convert values from 1D vector to 2D array with proper dimension
    double w1_arr[w1_r][w1_c];
    for ( int i=0; i<w1_r; i++ ) {
        for ( int j=0; j<w1_c; j++ ) {
            int a = w1_c*i + j;
            w1_arr[i][j] = w1_vec[a]; }
    }

    // convert 2D array to 2D vector
    for (int n = 0; n < w1_r; ++n)
        w1_mat.push_back(vector<double>(w1_arr[n], w1_arr[n]+w1_c));

    /*
    // print out w0
    cout << "w0 is: " << endl;
    for ( vector<vector<double> >::size_type i = 0; i < w0_mat.size(); i++ )
    {
       for ( vector<double>::size_type j = 0; j < w0_mat[i].size(); j++ )
       {
            cout << w0_mat[i][j] << ' ';
       }
       cout << endl;
    }

    // print out w1
    cout << "w1 is: " << endl;
    for ( vector<vector<double> >::size_type i = 0; i < w1_mat.size(); i++ )
    {
       for ( vector<double>::size_type j = 0; j < w1_mat[i].size(); j++ )
       {
            cout << w1_mat[i][j] << ' ';
       }
       cout << endl;
    } 
    */    
}



// Get 1D arrays (biases)
void ParseXML_1Darray(string& xmlFileName,
                    vector<double> & b0_vec, vector<double> & b1_vec)
{
    XMLReader r(xmlFileName, XMLReader::NONE);
    XMLBranch topB = r.GetTopBranch();

    // read variables from xml (as strings)
    string b0_str, b1_str;
    topB.GetChild("bias_0").GetData(b0_str);
    topB.GetChild("bias_1").GetData(b1_str);

    // create b0 matrix (1D vector) -------------------------------------------
    b0_vec = extractVaules(b0_str);
    int b0_r = 21; // row of b0 (number of neurons in the hidden layer)
    int b0_c = 1; // column of b0 (should be 1)

    // check if the total number of variables is correct
    unsigned int bvalues0 = b0_r * b0_c;
    if (b0_vec.size() != bvalues0) 
    { cout << "b0 has: " << b0_vec.size() << endl; log_fatal("b0 total value is wrong. Quitting !"); }
    
    // create b1 matrix (1D vector) -------------------------------------------
    b1_vec = extractVaules(b1_str);
    int b1_r = 3; // row of b1 (number of output classes)
    int b1_c = 1; // column of b1 (should be 1)

    // check if the total number of variables is correct
    unsigned int bvalues1 = b1_r * b1_c;
    if (b1_vec.size() != bvalues1) 
    { cout << "b1 has: " << b1_vec.size() << endl; log_fatal("b1 total value is wrong. Quitting !"); }

    // print out b0 and b1
    //for (vector<double>::iterator i = b1_vec.begin(); i != b1_vec.end(); ++i) cout << *i << endl;
    //for (vector<double>::iterator i = b0_vec.begin(); i != b0_vec.end(); ++i) cout << *i << endl;

}



// Do calculation like: activationFunc(X*W + B) = Y, 
// where X is input data and Y is output vector
vector<double> LinearSysCal(vector<double> data, 
                            vector< vector<double> > w_mat, 
                            vector<double> b_vec,
                            int option )
{
    int data_size = data.size();
    int w_size_r = w_mat.size();
    int w_size_c = w_mat[0].size();
    int out_size = b_vec.size();

    // create output in array
    double arr[out_size];

    // check dimensions
    if (data_size != w_size_r) {
        cout << data_size << " not equals to " << w_size_r << endl;
        log_fatal("Input dimension doesn't equal to weights row. Quitting !"); 
    }
    if (w_size_c != out_size) {
        cout << w_size_c << " not equals to " << out_size << endl;
        log_fatal("Weights column doesn't euqal to bias dimension. Quitting !"); 
    }

    // input data multiplies weights
    for ( int j = 0; j < out_size; j++)
    {
        arr[j] = 0;
        for (int i = 0; i < data_size; i++)
            arr[j] += data[i] * w_mat[i][j];
    }

    // add bias
    for ( int j = 0; j < out_size; j++)
    {
        arr[j] += b_vec[j];
    }

    // apply activation function
    if (option==0) {
        // reLU
        for ( int j = 0; j < out_size; j++)
        {
            if (arr[j]<0)
                arr[j] = 0.;
        }
    }
    else if (option==1) {
        // softmax
        double exp_sum = 0.;
        for ( int j = 0; j < out_size; j++)
        {
            arr[j] = exp(arr[j]);
            exp_sum += arr[j];
        }
        if (exp_sum > 0.)
            for ( int j = 0; j < out_size; j++)
            {
                arr[j] = (arr[j]/exp_sum);
            }
        else { log_fatal("Exponential sum of output vector is not positive. Quitting !"); }
    }
    else {
        log_fatal("Please provide proper option for activation function. Quitting !");
    }

    // convert array to vector
    vector<double> vec ( arr, arr + sizeof(arr)/sizeof(*arr) );

    return vec;
}



// Call this function to get max/min of one feature for one bin
void FieldStats(int bin, int index, double & fmin, double & fmax)
{
    // all features used for training
    string features[40] = {"nTankHit", "SFCFChi2", "planeChi2", "coreFitUnc", "zenithAngle",
                           "azimuthAngle", "coreFiduScale", "nHitSP10", "nHitSP20", "CxPE20",
                           "CxPE30", "CxPE40", "CxPE50", "CxPE40SPTime", "PINC", "GamCoreAge",
                           "numPoints", "scandelCore", "numSum", "scanedFrac", "fixedFrac",
                           "avePE", "nHit", "mPFnHits", "mPFnPlanes", "mPFp1nAssign",
                           "fAnnulusCharge0", "fAnnulusCharge1", "fAnnulusCharge2",
                           "fAnnulusCharge3", "fAnnulusCharge4", "fAnnulusCharge5",
                           "fAnnulusCharge6", "fAnnulusCharge7", "fAnnulusCharge8",
                           "disMax", "compactness", "nHit10ratio", "nHit20ratio", "nHitRatio"};

    // find out feature index
    //int index = std::distance(features, std::find(features, features + 40, feature));
    
    // return min and max for one feature in one bin
    double min = 0., max = 0.;

    // min and max from MC for each bin (the bin5 is a overflow bin with all high energy stuff)
    double min_0[40] = { 16.0, 23.11, 20.34, 1.5, 0.06709999999999999, -3.0809, 11.0, 22.0, 33.0, 2.44, 1.7400000000000002, 1.24, 0.8200000000000001, -8.600000000000001, 0.43000000000000005, -2.0, 0.0, 0.0, 0.0, 18.0, 0.0, 0.5, 35.0, 43.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.85, 0.1535971396166664, 0.3132530120481928, 0.48571428571428565, 0.45945945945945954 };
    double max_0[40] = { 54.0, 694.2264999999991, 114.42, 56.800000000000004, 0.9783000000000001, 3.0814000000000004, 147.0, 52.0, 55.0, 317.36649999999906, 285.42949999999723, 261.03000000000003, 238.64299999999815, 9.3, 5.81, 2.5, 14.0, 145.0, 5.0, 61.0, 50.0, 34.5, 88.0, 135.0, 3.0, 43.0, 0.87, 0.52, 0.35000000000000003, 0.45000000000000007, 0.49, 0.45000000000000007, 0.41000000000000003, 0.39, 0.38, 171.82, 32.41379310344827, 0.978723404255319, 1.0, 0.9454545454545453 };

    double min_1[40] = { 27.0, 38.32, 37.37, 1.4000000000000001, 0.07768400000000002, -3.0820000000000003, 11.0, 38.0, 55.0, 3.96, 3.02, 2.18, 1.51, -8.7, 0.63, -2.0, 0.0, 0.01, 0.0, 26.0, 0.0, 0.5, 57.0, 66.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.85, 0.1699498410998553, 0.3870967741935484, 0.5918367346938775, 0.5300719010641357 };
    double max_1[40] = { 68.0, 926.02, 153.15159999999915, 45.0, 0.9798, 3.0811, 147.0, 78.0, 83.0, 511.7679999999958, 445.6700000000001, 396.24, 355.0915999999992, 9.200000000000001, 6.07, 2.5, 19.0, 139.0, 7.0, 76.0, 66.0, 35.5, 114.0, 170.0, 3.0, 52.0, 0.8399999999999999, 0.51, 0.35000000000000003, 0.43, 0.46, 0.41999999999999993, 0.38, 0.36, 0.3400000000000001, 176.34, 30.566037735849058, 0.9692307692307692, 1.0, 0.9634146341463414 };

    double min_2[40] = { 40.0, 53.27, 58.27, 1.2000000000000002, 0.082, -3.0829, 12.0, 58.0, 80.0, 5.44, 4.18, 3.1311000000000058, 2.27, -8.6, 0.78, -2.0, 0.0, 0.01, 0.0, 35.0, 0.0, 0.5, 85.0, 94.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.85, 0.18444633563989615, 0.4642857142857143, 0.6950659352836881, 0.5930232558139535 };
    double max_2[40] = { 82.0, 1141.9689000000003, 192.26999999999998, 36.9, 0.9714, 3.0814000000000004, 147.0, 105.0, 111.0, 688.4057000000018, 587.8945000000007, 513.98, 454.18780000000027, 9.100000000000001, 6.1, 2.5, 23.0, 128.0, 10.0, 88.0, 78.0, 38.0, 139.0, 205.0, 3.0, 57.0, 0.8300000000000001, 0.51, 0.35000000000000003, 0.42, 0.44, 0.4, 0.36, 0.33, 0.31, 178.57890000000017, 30.33666666666672, 0.9626168224299065, 1.0, 0.9736842105263157 };

    double min_3[40] = { 54.0, 74.24, 85.7, 1.0, 0.0825, -3.0811, 12.0, 85.0, 110.0, 7.58, 5.82, 4.45, 3.39, -8.6, 0.93, -2.0, 0.0, 0.02, 0.0, 47.0, 0.0, 0.5, 114.0, 125.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.85, 0.21527148270739987, 0.535, 0.7708333333333334, 0.6755916547220895 };
    double max_3[40] = { 125.0, 1577.727200000002, 290.46, 28.400000000000002, 0.9560000000000001, 3.0815, 147.0, 204.0, 218.0, 983.3560000000102, 825.4980000000052, 707.143600000001, 609.163600000001, 9.1, 6.03, 2.5, 30.0, 109.0, 14.0, 99.0, 96.0, 46.5, 237.0, 286.0, 3.0, 60.0, 0.81, 0.54, 0.37, 0.41000000000000003, 0.41000000000000003, 0.37, 0.33, 0.3, 0.27, 182.94, 32.81493001555209, 0.9565217391304348, 0.9927536231884058, 0.9859154929577465 };

    double min_4[40] = { 101.0, 113.95, 173.64999999999998, 0.8, 0.0834, -3.0794, 12.0, 175.0, 214.0, 12.6, 9.49, 7.27, 5.67, -8.400000000000002, 1.1, -2.0, 0.0, 0.02, 0.0, 74.0, 0.0, 0.5, 225.0, 235.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.01, 0.01, 0.0, 0.0, 0.0, 0.0, 3.2, 0.2688218518248695, 0.6136363636363636, 0.8318934720673851, 0.7884615384615384 };
    double max_4[40] = { 170.0, 2208.9505999999997, 430.27239999999983, 18.8, 0.9424000000000001, 3.0803, 147.0, 308.0, 327.0, 1384.7491999999993, 1168.3043999999973, 989.9386, 841.7105999999994, 9.0, 5.77, 2.5, 35.0, 85.0, 20.0, 99.0, 99.0, 50.0, 353.0, 405.0, 3.0, 58.0, 0.77, 0.5700000000000001, 0.41999999999999993, 0.4, 0.38, 0.34, 0.29, 0.25, 0.21999999999999997, 186.94, 35.93016231005287, 0.9513274336283186, 0.9886363636363636, 0.9930313588850173 };

    double min_5[40] = { 145.0, 162.28, 270.15, 0.30000000000000004, 0.0789, -3.0804, 11.0, 280.0, 326.0, 20.38, 15.14, 11.4, 8.88, -8.2, 1.17, -2.0, 0.0, 0.0, 0.0, 92.0, 0.0, 0.5, 340.0, 350.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.01, 0.01, 0.01, 0.01, 0.0, 0.0, 11.26, 0.35022217194595706, 0.6974595842956121, 0.8793503480278422, 0.8789237668161435 };
    double max_5[40] = { 287.0, 4798.995900000002, 1089.9202000000002, 11.5, 0.9332, 3.0796, 146.0, 1005.0, 1021.0, 2260.4057000000007, 1901.6802000000002, 1629.6202000000003, 1397.9012000000012, 8.9, 5.31, 2.5, 38.0, 70.0, 26.0, 99.0, 99.0, 50.0, 1028.0, 1029.0, 3.0, 49.0, 0.7000000000000001, 0.51, 0.37, 0.34, 0.31, 0.27, 0.23, 0.19, 0.16, 191.18, 35.797844308856085, 0.9835543557635096, 0.9968847352024922, 1.0 };

    if (bin==0) { min = min_0[index]; max = max_0[index]; }
    if (bin==1) { min = min_1[index]; max = max_1[index]; }
    if (bin==2) { min = min_2[index]; max = max_2[index]; }
    if (bin==3) { min = min_3[index]; max = max_3[index]; }
    if (bin==4) { min = min_4[index]; max = max_4[index]; }
    if (bin==5) { min = min_5[index]; max = max_5[index]; }

    fmin = min;
    fmax = max;
}


int main(int argc, char** argv)
{
    CommandLineConfigurator cl(
    "Take in reconstructed xcd file and label each event.");

    cl.AddPositionalOption<string>("input",
    "Input XCDF file(s) without event label.");

    cl.AddOption<string>("model_0",
    "",
    "Trained model for bin 0.");

    cl.AddOption<string>("model_1",
    "",
    "Trained model for bin 1.");

    cl.AddOption<string>("model_2",
    "",
    "Trained model for bin 2.");

    cl.AddOption<string>("model_3",
    "",
    "Trained model for bin 3.");

    cl.AddOption<string>("model_4",
    "",
    "Trained model for bin 4.");

    cl.AddOption<string>("model_5",
    "",
    "Trained model for bin 5.");

    cl.AddOption<string>("output,o",
    ""
    "Output XCD file name");

    // Parse command line options
    log_info( "Processing I/O ***" );
    if (!cl.ParseCommandLine(argc, argv)) {
        // help();
        return 1;
    }

    // Check input XCD
    string input = cl.GetArgument<string>("input");
    if (input.empty()) {
        log_error("No input files specified. Quitting !");
        return 1;
    }

    // Check model files
    string model_0 = cl.GetArgument<string>("model_0");
    string model_1 = cl.GetArgument<string>("model_1");
    string model_2 = cl.GetArgument<string>("model_2");
    string model_3 = cl.GetArgument<string>("model_3");
    string model_4 = cl.GetArgument<string>("model_4");
    string model_5 = cl.GetArgument<string>("model_5");
    if (model_0.empty() || 
        model_1.empty() || 
        model_2.empty() || 
        model_3.empty() || 
        model_4.empty() ||
        model_5.empty() ) { 
        log_error("Not all model files are specified. Quitting !");
        return 1;
    }

    // Get name of output file
    string output = cl.GetArgument<string>("output");
    if (output.empty()) {
        log_error("No output file specified. Quitting !");
        return 1;
    }
  
    // I/O test
    cout << "Input files:" << input << endl;
    cout << "Output file: " << output << endl;
    log_info("I/O looks fine!");
  
    // Create output xcd file
    ofstream outf(output.c_str());
    XCDFFile outfile(outf);

    // the new field (class label)
    XCDFUnsignedIntegerField plabel = outfile.AllocateUnsignedIntegerField("rec.classLabel", 0);

    // Training features in array
    string features[40] = {"nTankHit", "SFCFChi2", "planeChi2", "coreFitUnc", "zenithAngle",
                           "azimuthAngle", "coreFiduScale", "nHitSP10", "nHitSP20", "CxPE20",
                           "CxPE30", "CxPE40", "CxPE50", "CxPE40SPTime", "PINC", "GamCoreAge",
                           "numPoints", "scandelCore", "numSum", "scanedFrac", "fixedFrac",
                           "avePE", "nHit", "mPFnHits", "mPFnPlanes", "mPFp1nAssign",
                           "fAnnulusCharge0", "fAnnulusCharge1", "fAnnulusCharge2",
                           "fAnnulusCharge3", "fAnnulusCharge4", "fAnnulusCharge5",
                           "fAnnulusCharge6", "fAnnulusCharge7", "fAnnulusCharge8",
                           "disMax", "compactness", "nHit10ratio", "nHit20ratio", "nHitRatio"};

    // Prepare output, copying all fields from the input files to the output
    FieldCopyBuffer buffer(outfile);

    // Load input xcd file ----------------------------------------------------
    log_info("Reading " << input);
    XCDFFile infile(input.c_str(), "r");

    // Check to see if we're using a REC file or not
    if (infile.HasField("rec.nChTot") && infile.HasField("rec.nChAvail")){ }
    else { log_fatal("Input file is not reconstructed xcdf !"); }

    XCDFUnsignedIntegerField nChTot_raw = infile.GetUnsignedIntegerField("rec.nChTot");
    XCDFUnsignedIntegerField nChAvail_raw = infile.GetUnsignedIntegerField("rec.nChAvail");
    XCDFUnsignedIntegerField nTankHit_raw = infile.GetUnsignedIntegerField("rec.nTankHit");
    XCDFUnsignedIntegerField coreFitStatus_raw = infile.GetUnsignedIntegerField("rec.coreFitStatus");
    XCDFUnsignedIntegerField angleFitStatus_raw = infile.GetUnsignedIntegerField("rec.angleFitStatus");
    XCDFFloatingPointField SFCFChi2_raw = infile.GetFloatingPointField("rec.SFCFChi2");
    XCDFFloatingPointField planeChi2_raw = infile.GetFloatingPointField("rec.planeChi2");
    XCDFFloatingPointField coreFitUnc_raw = infile.GetFloatingPointField("rec.coreFitUnc");
    XCDFFloatingPointField zenithAngle_raw = infile.GetFloatingPointField("rec.zenithAngle");
    XCDFFloatingPointField azimuthAngle_raw = infile.GetFloatingPointField("rec.azimuthAngle");
    XCDFUnsignedIntegerField coreFiduScale_raw = infile.GetUnsignedIntegerField("rec.coreFiduScale");
    XCDFUnsignedIntegerField nHitSP10_raw = infile.GetUnsignedIntegerField("rec.nHitSP10");
    XCDFUnsignedIntegerField nHitSP20_raw = infile.GetUnsignedIntegerField("rec.nHitSP20");
    XCDFFloatingPointField CxPE20_raw = infile.GetFloatingPointField("rec.CxPE20");
    XCDFFloatingPointField CxPE30_raw = infile.GetFloatingPointField("rec.CxPE30");
    XCDFFloatingPointField CxPE40_raw = infile.GetFloatingPointField("rec.CxPE40");
    XCDFFloatingPointField CxPE50_raw = infile.GetFloatingPointField("rec.CxPE50");
    XCDFFloatingPointField CxPE40SPTime_raw = infile.GetFloatingPointField("rec.CxPE40SPTime");
    XCDFFloatingPointField PINC_raw = infile.GetFloatingPointField("rec.PINC");
    XCDFFloatingPointField GamCoreAge_raw = infile.GetFloatingPointField("rec.GamCoreAge");
    XCDFFloatingPointField GamCorePackInt_raw = infile.GetFloatingPointField("rec.GamCorePackInt");
    XCDFFloatingPointField GamCoreChi2_raw = infile.GetFloatingPointField("rec.GamCoreChi2");
    XCDFUnsignedIntegerField nHit_raw = infile.GetUnsignedIntegerField("rec.nHit");
    XCDFFloatingPointField mPFnHits_raw = infile.GetFloatingPointField("rec.mPFnHits");
    XCDFFloatingPointField mPFnPlanes_raw = infile.GetFloatingPointField("rec.mPFnPlanes");
    XCDFFloatingPointField mPFp1nAssign_raw = infile.GetFloatingPointField("rec.mPFp1nAssign");
    XCDFFloatingPointField fAnnulusCharge0_raw = infile.GetFloatingPointField("rec.fAnnulusCharge0");
    XCDFFloatingPointField fAnnulusCharge1_raw = infile.GetFloatingPointField("rec.fAnnulusCharge1");
    XCDFFloatingPointField fAnnulusCharge2_raw = infile.GetFloatingPointField("rec.fAnnulusCharge2");
    XCDFFloatingPointField fAnnulusCharge3_raw = infile.GetFloatingPointField("rec.fAnnulusCharge3");
    XCDFFloatingPointField fAnnulusCharge4_raw = infile.GetFloatingPointField("rec.fAnnulusCharge4");
    XCDFFloatingPointField fAnnulusCharge5_raw = infile.GetFloatingPointField("rec.fAnnulusCharge5");
    XCDFFloatingPointField fAnnulusCharge6_raw = infile.GetFloatingPointField("rec.fAnnulusCharge6");
    XCDFFloatingPointField fAnnulusCharge7_raw = infile.GetFloatingPointField("rec.fAnnulusCharge7");
    XCDFFloatingPointField fAnnulusCharge8_raw = infile.GetFloatingPointField("rec.fAnnulusCharge8");
    XCDFFloatingPointField disMax_raw = infile.GetFloatingPointField("rec.disMax");

    // Copy over the XCDF fields
    set<string> fields;
    GetFieldNamesVisitor getFieldNamesVisitor(fields);
    infile.ApplyFieldVisitor(getFieldNamesVisitor);
    SelectFieldVisitor selectFieldVisitor(infile, fields, buffer);
    infile.ApplyFieldVisitor(selectFieldVisitor);

    // parse model xml files to have variable matices 
    log_info("Loading model: " << model_0);
    vector< vector<double> > w0_mat_0, w1_mat_0;
    ParseXML_2Darray(model_0, w0_mat_0, w1_mat_0);
    vector<double> b0_vec_0, b1_vec_0;
    ParseXML_1Darray(model_0, b0_vec_0, b1_vec_0);

    log_info("Loading model: " << model_1);
    vector< vector<double> > w0_mat_1, w1_mat_1;
    ParseXML_2Darray(model_1, w0_mat_1, w1_mat_1);
    vector<double> b0_vec_1, b1_vec_1;
    ParseXML_1Darray(model_1, b0_vec_1, b1_vec_1);

    log_info("Loading model: " << model_2);
    vector< vector<double> > w0_mat_2, w1_mat_2;
    ParseXML_2Darray(model_2, w0_mat_2, w1_mat_2);
    vector<double> b0_vec_2, b1_vec_2;
    ParseXML_1Darray(model_2, b0_vec_2, b1_vec_2);

    log_info("Loading model: " << model_3);
    vector< vector<double> > w0_mat_3, w1_mat_3;
    ParseXML_2Darray(model_3, w0_mat_3, w1_mat_3);
    vector<double> b0_vec_3, b1_vec_3;
    ParseXML_1Darray(model_3, b0_vec_3, b1_vec_3);

    log_info("Loading model: " << model_4);
    vector< vector<double> > w0_mat_4, w1_mat_4;
    ParseXML_2Darray(model_4, w0_mat_4, w1_mat_4);
    vector<double> b0_vec_4, b1_vec_4;
    ParseXML_1Darray(model_4, b0_vec_4, b1_vec_4);

    log_info("Loading model: " << model_5);
    vector< vector<double> > w0_mat_5, w1_mat_5;
    ParseXML_2Darray(model_5, w0_mat_5, w1_mat_5);
    vector<double> b0_vec_5, b1_vec_5;
    ParseXML_1Darray(model_5, b0_vec_5, b1_vec_5);

    // a map of models for all 6 bins
    map< int,vector< vector<double> > > w0Map;
    w0Map[0] = w0_mat_0; w0Map[1] = w0_mat_1; w0Map[2] = w0_mat_2;
    w0Map[3] = w0_mat_3; w0Map[4] = w0_mat_4; w0Map[5] = w0_mat_5;
    map< int,vector< vector<double> > > w1Map;
    w1Map[0] = w1_mat_0; w1Map[1] = w1_mat_1; w1Map[2] = w1_mat_2;
    w1Map[3] = w1_mat_3; w1Map[4] = w1_mat_4; w1Map[5] = w1_mat_5;
    map< int,vector<double> > b0Map;
    b0Map[0] = b0_vec_0; b0Map[1] = b0_vec_1; b0Map[2] = b0_vec_2;
    b0Map[3] = b0_vec_3; b0Map[4] = b0_vec_4; b0Map[5] = b0_vec_5;
    map< int,vector<double> > b1Map;
    b1Map[0] = b1_vec_0; b1Map[1] = b1_vec_1; b1Map[2] = b1_vec_2;
    b1Map[3] = b1_vec_3; b1Map[4] = b1_vec_4; b1Map[5] = b1_vec_5;

    // create a bunch of things before looping over events
    // a vector to hold feature values for each event
    vector<double> features_vec;
    // a map for each event to hold feature names and values
    //map<string,double> FeatureMap;
    // an array to hold normalized feature values for each event
    double features_arr[40];
    int bin_num;

    // events loop ------------------------------------------------------------
    int total_event_num = 0;
    int good_event_num = 0;
    while(infile.Read()){
        total_event_num ++;
        if ( (total_event_num%10000)==0 ) { log_info("Total event: " << total_event_num);  }

        // fields used for quality check and binning
        double nChTot = nChTot_raw.At(0); 
        double nChAvail = nChAvail_raw.At(0);
        double coreFitStatus = coreFitStatus_raw.At(0);
        double angleFitStatus = angleFitStatus_raw.At(0);
        double zenithAngle = zenithAngle_raw.At(0); 
        double coreFiduScale = coreFiduScale_raw.At(0); 
        double nHit = nHit_raw.At(0);

        // apply data quality cuts (reduce data size by 1/3) ------------------
        if ( nHit>25 && zenithAngle<1.05 && coreFiduScale<150 && coreFitStatus==0 
             && angleFitStatus==0 && nChAvail>=700 && nChAvail>0.9*nChTot )
        {
            good_event_num ++;

            // extract GamCore features
            double GamCorePackInt = GamCorePackInt_raw.At(0); 
            double numPoints = int(GamCorePackInt)/100000 ;
            double scandelCore = (int(GamCorePackInt)%100000)/100 ;
            double numSum = (int(GamCorePackInt)%100000)%100 ;
            double GamCoreChi2 = GamCoreChi2_raw.At(0);
            double scanedFrac = int(GamCoreChi2)/10000 ;
            double fixedFrac = (int(GamCoreChi2)%10000)/100 ;
            double avePE = ((int(GamCoreChi2)%10000)%100) / 2. + 0.5 ;

            // calculate cross features
            double mPFnHits = mPFnHits_raw.At(0);
            double nHitSP10 = nHitSP10_raw.At(0);
            double nHitSP20 = nHitSP20_raw.At(0);
            double CxPE40 = CxPE40_raw.At(0);
            double compactness = 0.;
            if (CxPE40!=0) {
                compactness = nHitSP20 / CxPE40 ;}
            double nHit10ratio = nHitSP10 / nHit ;
            double nHit20ratio = nHitSP20 / nHit ;
            double nHitRatio = 0.;
            if (mPFnHits!=0) {
                nHitRatio = nHit / mPFnHits; }
            
            // push feature values to a vector IN ORDER
            // all values use double format
            features_vec.clear();
            features_vec.push_back( double(nTankHit_raw.At(0)) );
            features_vec.push_back( double(SFCFChi2_raw.At(0)) );
            features_vec.push_back( double(planeChi2_raw.At(0)) );
            features_vec.push_back( double(coreFitUnc_raw.At(0)) );
            features_vec.push_back( double(zenithAngle_raw.At(0)) );
            features_vec.push_back( double(azimuthAngle_raw.At(0)) );
            features_vec.push_back( double(coreFiduScale_raw.At(0)) );
            features_vec.push_back( double(nHitSP10_raw.At(0)) );
            features_vec.push_back( double(nHitSP20_raw.At(0)) );
            features_vec.push_back( double(CxPE20_raw.At(0)) );
            features_vec.push_back( double(CxPE30_raw.At(0)) );
            features_vec.push_back( double(CxPE40_raw.At(0)) );
            features_vec.push_back( double(CxPE50_raw.At(0)) );
            features_vec.push_back( double(CxPE40SPTime_raw.At(0)) );
            features_vec.push_back( double(PINC_raw.At(0)) );
            features_vec.push_back( double(GamCoreAge_raw.At(0)) );
            features_vec.push_back( numPoints );
            features_vec.push_back( scandelCore );
            features_vec.push_back( numSum );
            features_vec.push_back( scanedFrac );
            features_vec.push_back( fixedFrac );
            features_vec.push_back( avePE );
            features_vec.push_back( double(nHit_raw.At(0)) );
            features_vec.push_back( double(mPFnHits_raw.At(0)) );
            features_vec.push_back( double(mPFnPlanes_raw.At(0)) );
            features_vec.push_back( double(mPFp1nAssign_raw.At(0)) );
            features_vec.push_back( double(fAnnulusCharge0_raw.At(0)) );
            features_vec.push_back( double(fAnnulusCharge1_raw.At(0)) );
            features_vec.push_back( double(fAnnulusCharge2_raw.At(0)) );
            features_vec.push_back( double(fAnnulusCharge3_raw.At(0)) );
            features_vec.push_back( double(fAnnulusCharge4_raw.At(0)) );
            features_vec.push_back( double(fAnnulusCharge5_raw.At(0)) );
            features_vec.push_back( double(fAnnulusCharge6_raw.At(0)) );
            features_vec.push_back( double(fAnnulusCharge7_raw.At(0)) );
            features_vec.push_back( double(fAnnulusCharge8_raw.At(0)) );
            features_vec.push_back( double(disMax_raw.At(0)) );
            features_vec.push_back( compactness );
            features_vec.push_back( nHit10ratio );
            features_vec.push_back( nHit20ratio );
            features_vec.push_back( nHitRatio );

            /*
            // fill feature map
            FeatureMap.clear();
            // loop over features
            for (int fi=0; fi<40; fi++) {  
                FeatureMap[features[fi]] = features_vec[fi];
                //cout << fi << ": " << features[fi] << ", " << features_vec[fi] << endl;
            }
            */

            // ******************************************************
            // Before assigning event to a bin, we have:
            // 1. features_vec: a vector of 40 feature values for each event.
            // 2. FeatureMap: a map of 40 feature names holding corresponding values for each event.
            // 3. features_arr[40]: an array to hold normalized feature values for each event.
            // ******************************************************
            
            // calculation based on bins
            // bin 0 (fHit 0a) --------------------------------------
            if      ( nHitSP20>0.030*nChAvail && nHitSP20<0.050*nChAvail ) { bin_num = 0; }
            // bin 1 (fHit 0b) --------------------------------------
            else if ( nHitSP20>0.050*nChAvail && nHitSP20<0.075*nChAvail ) { bin_num = 1; }
            // bin 2 (fHit 0c) --------------------------------------
            else if ( nHitSP20>0.075*nChAvail && nHitSP20<0.100*nChAvail ) { bin_num = 2; }
            // bin 3 (fHit 1) ---------------------------------------
            else if ( nHitSP20>0.100*nChAvail && nHitSP20<0.200*nChAvail ) { bin_num = 3; }
            // bin 4 (fHit 2) ---------------------------------------
            else if ( nHitSP20>0.200*nChAvail && nHitSP20<0.300*nChAvail ) { bin_num = 4; }
            // bin 5 (fHit H, overflow bin) -------------------------
            else if ( nHitSP20>0.300*nChAvail && nHitSP20<1.000*nChAvail ) { bin_num = 5; }
            // in case something is wrong
            else { bin_num = -1; plabel << int(9); }
    
            //cout << "Bin:" << bin_num << " ----------------------------------" << endl;
            // All calculation -----------------------------------------------------
            if (bin_num != -1){
                // loop over features for normalization
                double feature_min, feature_max;
                for (int fi=0; fi<40; fi++) {
                    FieldStats(bin_num, fi, feature_min, feature_max);
                    //cout << fi << "|" << features[fi] << "|" << features_vec[fi] << "|" << feature_min << "|" << feature_max << endl;
                    features_arr[fi] = double(features_vec[fi] - feature_min) / (feature_max - feature_min);
                    if (features_arr[fi]<0.) { features_arr[fi]=0.; } // values should be in (0,1)
                    if (features_arr[fi]>1.) { features_arr[fi]=1.; } }

                // pass event to model
                vector<double> features_vec_normalized ( features_arr, features_arr + sizeof(features_arr)/sizeof(*features_arr) );
                vector<double> res0_vec = LinearSysCal(features_vec_normalized, w0Map[bin_num], b0Map[bin_num], int(0) );
                vector<double> res1_vec = LinearSysCal(res0_vec, w1Map[bin_num], b1Map[bin_num], int(1) );
                //for (vector<double>::iterator i = res0_vec.begin(); i != res0_vec.end(); ++i) cout << *i << endl;
                //for (vector<double>::iterator i = res1_vec.begin(); i != res1_vec.end(); ++i) cout << *i << endl;

                // calculate label
                if (res1_vec.size()!=3) { log_fatal("Output class number is NOT 3 !"); }
                double res1_arr[3]; // use array for calculation
                for (int i=0;i<3;i++) { res1_arr[i] = res1_vec[i]; }
                //double max_prob = *max_element( res1_arr, res1_arr + sizeof(res1_arr)/sizeof(*res1_arr) );
                int classnum = std::distance(res1_arr, max_element(res1_arr, res1_arr + sizeof(res1_arr)/sizeof(*res1_arr) ) );
                //cout << max_prob << "|" << classnum << endl;

                /*
                // debug outputs
                cout << "debug outputs" << endl;
                //if (classnum==0) {
                    for (int fi=0; fi<40; fi++) { cout << features_vec_normalized[fi] << ", "; } cout << endl;
                    for (vector<double>::iterator i = res1_vec.begin(); i != res1_vec.end(); ++i) cout << *i << endl;
                    cout << classnum << endl;
                    cout << "***************************" << endl;
                //}
                */

                plabel << classnum;
            } // end of calculation

        } // end of events after quality check

        // bad events (failed quality cuts)
        else { plabel << int(9); };

        // Copy data into the file copy buffer and write output
        buffer.CopyData();
        outfile.Write();

    } // end of events loop

    // Copy over the header from input to output file
    CopyComments(infile, outfile);
    
    infile.Close();
    log_info("Writing file to: " << output);
    outfile.Close();

    cout << "Events total/quality: " << total_event_num << "/" << good_event_num << endl;

    return 0;
}


