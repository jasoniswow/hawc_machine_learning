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
    int w0_c = 22; // column of w0 (number of neurons in the hidden layer)

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
    int w1_r = 22; // row of w1 (number of neurons in the hidden layer)
    int w1_c = 4; // column of w1 (number of output classes)

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
    int b0_r = 22; // row of b0 (number of neurons in the hidden layer)
    int b0_c = 1; // column of b0 (should be 1)

    // check if the total number of variables is correct
    unsigned int bvalues0 = b0_r * b0_c;
    if (b0_vec.size() != bvalues0) 
    { cout << "b0 has: " << b0_vec.size() << endl; log_fatal("b0 total value is wrong. Quitting !"); }
    
    // create b1 matrix (1D vector) -------------------------------------------
    b1_vec = extractVaules(b1_str);
    int b1_r = 4; // row of b1 (number of output classes)
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
double FieldStats(string file, int bin, int index)
{
    XCDFFile infile(file.c_str(), "r");

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
    double min_0[40] = { 10, 1.37, 3.0300000000000002, 0.0, 0.0001, -3.1416, 0, 7, 33, 0.0, 0.0, 0.0, 0.0, -10.0, 0.04, -2.0, 0, 0.0, 0, 8, 0, 0.5, 33, 34.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.09411764705882353, 0.2360248447204969, 0.3 };
    double max_0[40] = { 112, 2022.0700000000002, 222.77, 200.0, 1.0499, 3.1416000000000004, 149, 56, 56, 8078.83, 8078.83, 8078.83, 8078.83, 10.000000000000002, 25.16, 2.5, 26, 236.0, 16, 88, 87, 50.0, 186, 396.0, 3.0, 167.0, 1.0, 1.0, 0.98, 0.99, 0.98, 0.99, 0.98, 0.98, 0.97, 202.44, 196.15384615384616, 1.0, 1.0, 1.0 };

    double min_1[40] = { 18, 17.58, 10.83, 0.0, 0.0005, -3.1416, 0, 19, 55, 1.28, 0.0, 0.0, 0.0, -10.0, 0.19, -2.0, 0, 0.0, 0, 15, 0, 0.5, 55, 57.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.12903225806451613, 0.26851851851851855, 0.3142857142857143 };
    double max_1[40] = { 146, 2451.7400000000002, 286.19, 200.0, 1.0499, 3.1416000000000004, 149, 84, 84, 6618.68, 6618.68, 6618.68, 6618.68, 10.000000000000002, 17.8, 2.5, 30, 226.0, 28, 97, 96, 50.0, 249, 574.0, 3.0, 211.0, 0.99, 0.98, 0.95, 0.98, 0.98, 0.97, 0.96, 0.96, 0.96, 202.44, 183.33333333333334, 1.0, 1.0, 1.0 };

    double min_2[40] = { 28, 24.73, 26.03, 0.0, 0.0004, -3.1416, 0, 36, 82, 2.52, 1.47, 0.0, 0.0, -10.0, 0.32, -2.0, 0, 0.0, 0, 24, 0, 0.5, 82, 86.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1510791366906475, 0.3110367892976589, 0.33238636363636365 };
    double max_2[40] = { 166, 2915.46, 368.01, 200.0, 1.0499, 3.1416, 149, 112, 112, 6283.25, 4724.67, 4424.11, 4424.11, 10.000000000000002, 14.85, 2.5, 35, 227.0, 32, 99, 99, 50.0, 325, 590.0, 3.0, 211.0, 0.98, 0.97, 0.96, 0.96, 0.96, 0.96, 0.94, 0.96, 0.95, 202.44, 93.22033898305085, 1.0, 1.0, 1.0 };

    double min_3[40] = { 37, 33.690000000000005, 46.99, 0.0, 0.0007, -3.1416, 0, 47, 109, 3.04, 2.13, 1.18, 0.88, -10.0, 0.45, -2.0, 0, 0.0, 0, 29, 0, 0.5, 111, 113.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.02219456438045581, 0.1529051987767584, 0.3178082191780822, 0.3578199052132701 };
    double max_3[40] = { 210, 3969.71, 561.39, 200.0, 1.0499, 3.1416, 149, 220, 224, 7351.03, 7351.03, 7351.03, 7351.03, 10.000000000000002, 13.01, 2.5, 46, 224.0, 50, 99, 99, 50.0, 481, 763.0, 3.0, 309.0, 0.97, 0.96, 0.94, 0.96, 0.94, 0.94, 0.9100000000000001, 0.9100000000000001, 0.9299999999999999, 202.44, 95.76271186440678, 1.0, 1.0, 1.0 };

    double min_4[40] = { 72, 53.26, 104.85000000000001, 0.0, 0.0007, -3.1416, 0, 106, 213, 5.33, 4.69, 3.13, 2.37, -10.0, 0.72, -2.0, 0, 0.0, 0, 35, 0, 0.5, 220, 222.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.04686503637802413, 0.30129390018484287, 0.5010799136069114, 0.37516170763260026 };
    double max_4[40] = { 229, 4497.88, 749.53, 200.0, 1.0499, 3.1416000000000004, 149, 328, 335, 5930.02, 5930.02, 5632.1, 5632.1, 10.000000000000002, 11.44, 2.5, 50, 189.0, 53, 99, 99, 50.0, 541, 834.0, 3.0, 318.0, 0.94, 0.9100000000000001, 0.8900000000000001, 0.9100000000000001, 0.8700000000000001, 0.8400000000000001, 0.8400000000000001, 0.8600000000000001, 0.8300000000000001, 202.44, 76.32398753894081, 0.9963369963369964, 1.0, 1.0 };

    double min_5[40] = { 99, 89.55000000000001, 175.67000000000002, 0.0, 0.0007, -3.1416, 0, 187, 317, 8.68, 7.29, 5.16, 3.3000000000000003, -10.0, 0.8200000000000001, -2.0, 0, 0.0, 0, 54, 0, 0.5, 321, 331.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.07589360087370194, 0.3772893772893773, 0.6080402010050251, 0.45569620253164556 };
    double max_5[40] = { 289, 11595.15, 1494.84, 130.3, 1.0499, 3.1416, 149, 1077, 1083, 9947.880000000001, 8087.08, 8087.08, 8087.08, 10.000000000000002, 8.51, 2.5, 56, 170.0, 76, 99, 99, 50.0, 1087, 1087.0, 3.0, 374.0, 0.9000000000000001, 0.8400000000000001, 0.8300000000000001, 0.8200000000000001, 0.8400000000000001, 0.8400000000000001, 0.73, 0.75, 0.74, 202.44, 71.45454545454545, 1.0, 1.0, 1.0 };

    if (bin==0) { min = min_0[index]; max = max_0[index]; }
    if (bin==1) { min = min_1[index]; max = max_1[index]; }
    if (bin==2) { min = min_2[index]; max = max_2[index]; }
    if (bin==3) { min = min_3[index]; max = max_3[index]; }
    if (bin==4) { min = min_4[index]; max = max_4[index]; }
    if (bin==5) { min = min_5[index]; max = max_5[index]; }

    return (min,max);
}

//void testFunc();

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

    /*
    // some tests
    log_info("Testing multiplication: ");
    double t_data_arr[4] = { 1., 2., 3., 4. };
    double t_w_arr[4][2] = { {1., -2.}, {3., -4.}, {-5., 6.}, {7., 8.} };
    double t_b_arr[2] = { 0.1, -2.0 };
    vector<double> t_data_vec ( t_data_arr, t_data_arr + sizeof(t_data_arr)/sizeof(*t_data_arr) );
    vector<double> t_b_vec ( t_b_arr, t_b_arr + sizeof(t_b_arr)/sizeof(*t_b_arr) );
    vector< vector<double> > t_w_vec;
    for (int n = 0; n < 4; ++n)
        t_w_vec.push_back(vector<double>(t_w_arr[n], t_w_arr[n]+2));
    int option = 1;
    vector<double> res_vec = LinearSysCal(t_data_vec, t_w_vec, t_b_vec, option );
    for (vector<double>::iterator i = res_vec.begin(); i != res_vec.end(); ++i) cout << *i << endl;
    */

    // create a bunch of things before looping over events
    // a vector to hold feature values for each event
    vector<double> features_vec;
    // a map for each event to hold feature names and values
    map<string,double> FeatureMap;
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
        double nHit = nHit_raw.At(0);
        double mPFnHits = mPFnHits_raw.At(0);
        double nHitSP10 = nHitSP10_raw.At(0);
        double nHitSP20 = nHitSP20_raw.At(0);
        double CxPE40 = CxPE40_raw.At(0);
        double compactness = 0.;
        if (CxPE40!=0) {
            double compactness = double(nHitSP20) / CxPE40 ;}
        double nHit10ratio = double(nHitSP10) / nHit ;
        double nHit20ratio = double(nHitSP20) / nHit ;
        double nHitRatio = 0.;
        if (mPFnHits!=0) {
            double nHitRatio = double(nHit) / mPFnHits ;}

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

        // apply data quality cuts (reduce data size by 1/3) ------------------
        if ( nHit>25 && zenithAngle<1.05 && coreFiduScale<150 && coreFitStatus==0 
             && angleFitStatus==0 && nChAvail>=700 && nChAvail>0.9*nChTot )
        {
            good_event_num ++;

            // fill feature map
            FeatureMap.clear();
            // loop over features
            for (int fi=0; fi<40; fi++) {  
                FeatureMap[features[fi]] = features_vec[fi];
                //cout << fi << ": " << features[fi] << ", " << features_vec[fi] << endl;
            }

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

            // All calculation -----------------------------------------------------
            if (bin_num != -1){
                // loop over features for normalization
                double feature_min, feature_max;
                for (int fi=0; fi<40; fi++) {
                    feature_min, feature_max = FieldStats(input, bin_num, fi);
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
                if (res1_vec.size()!=4) { log_fatal("Output class number is NOT 4 !"); }
                double res1_arr[4]; // use array for calculation
                for (int i=0;i<4;i++) { res1_arr[i] = res1_vec[i]; }
                //double max_prob = *max_element( res1_arr, res1_arr + sizeof(res1_arr)/sizeof(*res1_arr) );
                int classnum = std::distance(res1_arr, max_element(res1_arr, res1_arr + sizeof(res1_arr)/sizeof(*res1_arr) ) );
                //cout << max_prob << "|" << classnum << endl;
                plabel << classnum;
            } // end of calculation

        }
        // bad events (failed quality cuts)
        else { plabel << int(9); };

        // Copy data into the file copy buffer and write output
        buffer.CopyData();
        outfile.Write();

    } // end of file reading (events loop)

    // Copy over the header from input to output file
    CopyComments(infile, outfile);
    
    infile.Close();
    log_info("Writing file to: " << output);
    outfile.Close();

    cout << "Events total/good: " << total_event_num << "/" << good_event_num << endl;

    return 0;
}



// small test of array/vector operations
void testFunc()
{

    // the way to have the size of an (int) array can be:
    // "sizeof(arr)/sizeof(*arr)" or  "sizeof(arr)/sizeof(arr[0])" or "sizeof(arr)/sizeof(int)"

    cout << "1D array -------------------------------------------" << endl;
    int arr[4];
    for (int i=0;i<4;i++) {
        arr[i] = i+5;
    }

    cout << "one way to print -----------------------------------" << endl;
    for ( unsigned int a = 0; a < sizeof(arr)/sizeof(*arr); a = a + 1 ) {
        //cout << arr[a] << endl; 
    }

    cout << "another way to have 1D array -----------------------" << endl;
    static const int arr1[] = {1,2,3,4,5,6,7,8};

    cout << "1D vector (normally it's good to use an array to construct a vector)" << endl;
    std::vector<int> vec;
    for (int i=0;i<4;i++) {
        vec.push_back(i);
        //cout << vec[i] << endl;
    }

    cout << "a better way to print ------------------------------" << endl;
    //for (std::vector<int>::iterator i = vec.begin(); i != vec.end(); ++i) cout << *i << endl;

    cout << "create 1D vector from 1D array ---------------------" << endl;
    std::vector<int> vec1 ( arr1, arr1 + sizeof(arr1)/sizeof(*arr1) );

    cout << "create a 2D array from 1D array --------------------" << endl;
    int arr2[4][2];
    for (int i=0;i<4;i++) {
        for (int j=0;j<2;j++) {
            int a = 2*i + j;
            arr2[i][j] = arr1[a]; 
        }
    }

    cout << "create a 2D vector (matrix) from a 2D array --------" << endl;
    std::vector< std::vector<int> > mat2;
    for (int n = 0; n < 4; ++n)
        mat2.push_back(std::vector<int>(arr2[n], arr2[n]+2));

    cout << "print 2D vector ------------------------------------" << endl;
    for ( std::vector<std::vector<int> >::size_type i=0; i<mat2.size(); i++ )
    {
       for ( std::vector<int>::size_type j=0; j<mat2[i].size(); j++ )
       {
          //cout << mat2[i][j] << ' ';
       }
       //cout << std::endl;
    }        

    cout << "summation of two vectors ---------------------------" << endl;
    static const int aarr[] = {1,2,3,4};
    static const int barr[] = {5,6,7,8};
    std::vector<int> avec (aarr, aarr + sizeof(aarr)/sizeof(*aarr) );
    std::vector<int> bvec (barr, barr + sizeof(barr)/sizeof(*barr) );
    std::vector<int> sum1(avec.size()), sum2;
    // one way of doing summation
    std::transform (avec.begin(), avec.end(), bvec.begin(), sum1.begin(), std::plus<int>());
    //for (std::vector<int>::iterator i = sum1.begin(); i != sum1.end(); ++i) cout << *i << endl;
    // another way
    for (unsigned int i=0; i<avec.size(); i++) {
        sum2.push_back( avec[i] + bvec[i] );
        //cout << sum2[i] << endl;
    }

    cout << "element-wise multiplication of two vectors ---------" << endl;
    std:vector<int> emul(avec.size());
    std::transform (avec.begin(), avec.end(), bvec.begin(), emul.begin(), std::multiplies<int>());
    //for (std::vector<int>::iterator i = emul.begin(); i != emul.end(); ++i) cout << *i << endl;

}





