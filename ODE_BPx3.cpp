#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>
#include <random>
using namespace std;

#define INNODE 1
#define HIDENODE 4
#define HIDENODE_2 4
#define HIDENODE_3 4
#define OUTNODE 1

double rate = 0.5;
double threshold = 1e-8;    //最小误差
size_t mosttimes = 1e6;     //最大迭代次数

//样本类
struct Sample {
    vector<double> in, out;
};

//神经元
struct Node {
    double value{}, bias{}, bias_delta{};
    vector<double> weight, weight_delta;
};

namespace utils {

    inline double sigmoid(double x) {
        double res = 1.0 / (1.0 + exp(-x));
        return res;
    }

    //读取文件
    vector<double> getFileData(string filename) {
        vector<double> res;
        ifstream in(filename);
        if (in.is_open()) {
            while (!in.eof()) {
                double buffer;
                in >> buffer;
                res.push_back(buffer);
            }
            in.close();
        }
        else {
            cout << "Error in reading " << filename << endl;
        }
        return res;
    }

    //获得训练数据
    vector<Sample> getTrainData(string filename) {
        vector<Sample> res;
        vector<double> buffer;
        buffer = getFileData(filename);

        for (size_t i = 0; i < buffer.size(); i += INNODE + OUTNODE) {
            Sample tmp;
            for (size_t t = 0; t < INNODE; t++) {
                tmp.in.push_back(buffer[i + t]);
            }
            for (size_t t = 0; t < OUTNODE; t++) {
                tmp.out.push_back(buffer[i + INNODE + t]);
            }
            res.push_back(tmp);
        }
        return res;
    }

    //获得测试数据
    vector<Sample> getTestData(string filename) {
        vector<Sample> res;

        vector<double> buffer = getFileData(filename);

        for (size_t i = 0; i < buffer.size(); i += INNODE) {
            Sample tmp;
            for (size_t t = 0; t < INNODE; t++) {
                tmp.in.push_back(buffer[i + t]);
            }
            res.push_back(tmp);
        }
        return res;
    }

}

Node* inputLayer[INNODE], * hideLayer[HIDENODE], * hideLayer_2[HIDENODE_2], * hideLayer_3[HIDENODE_3], * outLayer[OUTNODE];

inline void init() {

    mt19937 rd;
    rd.seed(random_device()());

    uniform_real_distribution<double> distribution(-1, 1);

    for (size_t i = 0; i < INNODE; i++) {
        ::inputLayer[i] = new Node();
        for (size_t j = 0; j < HIDENODE; j++) {
            ::inputLayer[i]->weight.push_back(distribution(rd));
            ::inputLayer[i]->weight_delta.push_back(0.f);
        }
    }

    for (size_t i = 0; i < HIDENODE; i++) {
        ::hideLayer[i] = new Node();
        ::hideLayer[i]->bias = distribution(rd);
        for (size_t j = 0; j < HIDENODE_2; j++) {
            ::hideLayer[i]->weight.push_back(distribution(rd));
            ::hideLayer[i]->weight_delta.push_back(0.f);
        }
    }

    for (size_t i = 0; i < HIDENODE_2; i++) {
        ::hideLayer_2[i] = new Node();
        ::hideLayer_2[i]->bias = distribution(rd);
        for (size_t j = 0; j < HIDENODE_3; j++) {
            ::hideLayer_2[i]->weight.push_back(distribution(rd));
            ::hideLayer_2[i]->weight_delta.push_back(0.f);
        }
    }

    for (size_t i = 0; i < HIDENODE_3; i++) {
        ::hideLayer_3[i] = new Node();
        ::hideLayer_3[i]->bias = distribution(rd);
        for (size_t j = 0; j < OUTNODE; j++) {
            ::hideLayer_3[i]->weight.push_back(distribution(rd));
            ::hideLayer_3[i]->weight_delta.push_back(0.f);
        }
    }

    for (size_t i = 0; i < OUTNODE; i++) {
        ::outLayer[i] = new Node();
        ::outLayer[i]->bias = distribution(rd);
    }

}

inline void reset_delta() {

    for (size_t i = 0; i < INNODE; i++) {
        ::inputLayer[i]->weight_delta.assign(::inputLayer[i]->weight_delta.size(), 0.f);
    }

    for (size_t i = 0; i < HIDENODE; i++) {
        ::hideLayer[i]->bias_delta = 0.f;
        ::hideLayer[i]->weight_delta.assign(::hideLayer[i]->weight_delta.size(), 0.f);
    }

    for (size_t i = 0; i < HIDENODE_2; i++) {
        ::hideLayer_2[i]->bias_delta = 0.f;
        ::hideLayer_2[i]->weight_delta.assign(::hideLayer_2[i]->weight_delta.size(), 0.f);
    }

    for (size_t i = 0; i < HIDENODE_3; i++) {
        ::hideLayer_3[i]->bias_delta = 0.f;
        ::hideLayer_3[i]->weight_delta.assign(::hideLayer_3[i]->weight_delta.size(), 0.f);
    }

    for (size_t i = 0; i < OUTNODE; i++) {
        ::outLayer[i]->bias_delta = 0.f;
    }

}

int main() {
    init();

    vector<Sample> train_data = utils::getTrainData("traindata.txt");

    for (size_t times = 0; times < mosttimes; times++) {

        reset_delta();

        double error_max = 0.f;

        for (auto& idx : train_data) {
            for (size_t i = 0; i < INNODE; i++) {
                ::inputLayer[i]->value = idx.in[i];
            }

            //正向传播
            for (size_t j = 0; j < HIDENODE; j++) {
                double sum = 0;
                for (size_t i = 0; i < INNODE; i++) {
                    sum += ::inputLayer[i]->value * ::inputLayer[i]->weight[j];
                }
                sum -= ::hideLayer[j]->bias;

                ::hideLayer[j]->value = utils::sigmoid(sum);
            }

            for (size_t j = 0; j < HIDENODE_2; j++) {
                double sum = 0;
                for (size_t i = 0; i < HIDENODE; i++) {
                    sum += ::hideLayer[i]->value * ::hideLayer[i]->weight[j];
                }
                sum -= ::hideLayer_2[j]->bias;

                ::hideLayer_2[j]->value = utils::sigmoid(sum);
            }

            for (size_t j = 0; j < HIDENODE_3; j++) {
                double sum = 0;
                for (size_t i = 0; i < HIDENODE_2; i++) {
                    sum += ::hideLayer_2[i]->value * ::hideLayer_2[i]->weight[j];
                }
                sum -= ::hideLayer_3[j]->bias;

                ::hideLayer_3[j]->value = utils::sigmoid(sum);
            }

            for (size_t j = 0; j < OUTNODE; j++) {
                double sum = 0;
                for (size_t i = 0; i < HIDENODE_3; i++) {
                    sum += ::hideLayer_3[i]->value * ::hideLayer_3[i]->weight[j];
                }
                sum -= ::outLayer[j]->bias;

                ::outLayer[j]->value = sum;
            }

            //计算误差
            double error = 0.f;
            for (size_t i = 0; i < OUTNODE; i++) {
                double tmp = fabs(::outLayer[i]->value - idx.out[i]);
                error += tmp * tmp / 2;
            }

            error_max = max(error_max, error);

            //反向传播


            for (size_t n = 0; n < OUTNODE; n++) {
                double bias_delta = -(idx.out[n] - ::outLayer[n]->value);
                ::outLayer[n]->bias_delta += bias_delta;
            }

            for (size_t k = 0; k < HIDENODE_3; k++) {
                for (size_t n = 0; n < OUTNODE; n++) {
                    double weight_delta = (idx.out[n] - ::outLayer[n]->value) * ::hideLayer_3[k]->value;
                    ::hideLayer_3[k]->weight_delta[n] += weight_delta;
                }
            }

            for (size_t k = 0; k < HIDENODE_3; k++) {
                double sum = 0;
                for (size_t n = 0; n < OUTNODE; n++) {
                    sum += -(idx.out[n] - ::outLayer[n]->value) * ::hideLayer_3[k]->weight[n] *
                        ::hideLayer_3[k]->value * (1.0 - ::hideLayer_3[k]->value);
                }
                ::hideLayer_3[k]->bias_delta += sum ;
            }

            for (size_t j = 0; j < HIDENODE_2; j++) {
                for (size_t k = 0; k < HIDENODE_3; k++) {
                    double sum = 0.f;
                    for (size_t n = 0; n < OUTNODE; n++) {
                        sum += (idx.out[n] - ::outLayer[n]->value) * hideLayer_3[k]->weight[n] *
                            ::hideLayer_3[k]->value * (1.0 - hideLayer_3[k]->value) * ::hideLayer_2[j]->value;
                    }
                    ::hideLayer_2[j]->weight_delta[k] += sum;
                }
            }

            //vvvvvvvvvvvvvvvvvvvvvvvvvv

            for (size_t j = 0; j < HIDENODE_2; j++) {
                double sum = 0.f;
                for (size_t k = 0; k < HIDENODE_3; k++) {
                    for (size_t n = 0; n < OUTNODE; n++) {
                        sum += -(idx.out[n] - ::outLayer[n]->value) * hideLayer_3[k]->weight[n] *
                            ::hideLayer_3[k]->value * (1.0 - hideLayer_3[k]->value) * ::hideLayer_2[j]->weight[k] *
                            ::hideLayer_2[j]->value * (1.0 - hideLayer_2[j]->value);
                    }
                }
                ::hideLayer_2[j]->bias_delta += sum;
            }

            for (size_t i = 0; i < HIDENODE; i++) {
                for (size_t j = 0; j < HIDENODE_2; j++) {
                    double sum = 0.f;
                    for (size_t k = 0; k < HIDENODE_3; k++) {
                        for (size_t n = 0; n < OUTNODE; n++) {
                            sum += (idx.out[n] - ::outLayer[n]->value) * hideLayer_3[k]->weight[n] *
                                ::hideLayer_3[k]->value * (1.0 - hideLayer_3[k]->value) * ::hideLayer_2[j]->weight[k] *
                                ::hideLayer_2[j]->value * (1.0 - hideLayer_2[j]->value) * ::hideLayer[i]->value;
                        }
                    }
                    ::hideLayer[i]->weight_delta[j] += sum;
                }
            }

            for (size_t i = 0; i < HIDENODE; i++) {
                double sum = 0.f;
                for (size_t j = 0; j < HIDENODE_2; j++) {
                    for (size_t k = 0; k < HIDENODE_3; k++) {
                        for (size_t n = 0; n < OUTNODE; n++) {
                            sum += -(idx.out[n] - ::outLayer[n]->value) * hideLayer_3[k]->weight[n] *
                                ::hideLayer_3[k]->value * (1.0 - hideLayer_3[k]->value) * ::hideLayer_2[j]->weight[k] *
                                ::hideLayer_2[j]->value * (1.0 - hideLayer_2[j]->value) * ::hideLayer[i]->weight[j] *
                                ::hideLayer[i]->value * (1 - ::hideLayer[i]->value);
                        }
                    }
                    ::hideLayer[i]->bias_delta += sum;
                }
            }

            for (size_t m = 0; m < INNODE; m++) {
                for (size_t i = 0; i < HIDENODE; i++) {
                    double sum = 0.f;
                    for (size_t j = 0; j < HIDENODE_2; j++) {
                        for (size_t k = 0; k < HIDENODE_3; k++) {
                            for (size_t n = 0; n < OUTNODE; n++) {
                                sum += (idx.out[n] - ::outLayer[n]->value) * hideLayer_3[k]->weight[n] *
                                    ::hideLayer_3[k]->value * (1.0 - hideLayer_3[k]->value) * ::hideLayer_2[j]->weight[k] *
                                    ::hideLayer_2[j]->value * (1.0 - hideLayer_2[j]->value) * ::hideLayer[i]->weight[j] *
                                    ::hideLayer[i]->value * (1 - ::hideLayer[i]->value) * inputLayer[m]->value;
                            }
                        }
                    }
                    ::inputLayer[m]->weight_delta[i] += sum;
                }
            }

        }

        if (error_max < ::threshold) {
            cout << "Success with " << times + 1 << "times training." << endl;
            cout << "Maximum error: " << error_max << endl;
            break;
        }

        auto train_data_size = double(train_data.size());

        for (size_t i = 0; i < INNODE; i++) {
            for (size_t j = 0; j < HIDENODE; j++) {
                ::inputLayer[i]->weight[j] += rate * ::inputLayer[i]->weight_delta[j] / train_data_size;
            }
        }

        for (size_t i = 0; i < HIDENODE; i++) {
            ::hideLayer[i]->bias +=
                rate * ::hideLayer[i]->bias_delta / train_data_size;
            for (size_t j = 0; j < HIDENODE_2; j++) {
                ::hideLayer[i]->weight[j] +=
                    rate * ::hideLayer[i]->weight_delta[j] / train_data_size;
            }
        }

        for (size_t i = 0; i < HIDENODE_2; i++) {
            ::hideLayer_2[i]->bias +=
                rate * ::hideLayer_2[i]->bias_delta / train_data_size;
            for (size_t j = 0; j < HIDENODE_3; j++) {
                ::hideLayer_2[i]->weight[j] +=
                    rate * ::hideLayer_2[i]->weight_delta[j] / train_data_size;
            }
        }

        for (size_t i = 0; i < HIDENODE_3; i++) {
            ::hideLayer_3[i]->bias +=
                rate * ::hideLayer_3[i]->bias_delta / train_data_size;
            for (size_t j = 0; j < OUTNODE; j++) {
                ::hideLayer_3[i]->weight[j] +=
                    rate * ::hideLayer_3[i]->weight_delta[j] / train_data_size;
            }
        }

        for (size_t i = 0; i < OUTNODE; i++) {
            ::outLayer[i]->bias +=
                rate * ::outLayer[i]->bias_delta / train_data_size;
        }


    }
    //预测
    vector<Sample> test_data = utils::getTestData("testdata.txt");
    for (auto& idx : test_data) {

        for (size_t m = 0; m < INNODE; m++) {
            ::inputLayer[m]->value = idx.in[m];
        }

        for (size_t i = 0; i < HIDENODE; i++) {
            double sum = 0;
            for (size_t m = 0; m < INNODE; m++) {
                sum += ::inputLayer[m]->value * inputLayer[m]->weight[i];
            }
            sum -= ::hideLayer[i]->bias;

            ::hideLayer[i]->value = utils::sigmoid(sum);
        }

        for (size_t j = 0; j < HIDENODE_2; j++) {
            double sum = 0;
            for (size_t i = 0; i < HIDENODE; i++) {
                sum += ::hideLayer[i]->value * hideLayer[i]->weight[j];
            }
            sum -= ::hideLayer_2[j]->bias;

            ::hideLayer_2[j]->value = utils::sigmoid(sum);
        }

        for (size_t k = 0; k < HIDENODE_3; k++) {
            double sum = 0;
            for (size_t j = 0; j < HIDENODE_2; j++) {
                sum += ::hideLayer_2[j]->value * hideLayer_2[j]->weight[k];
            }
            sum -= ::hideLayer_3[k]->bias;
            ::hideLayer_3[k]->value = utils::sigmoid(sum);
        }

        for (size_t n = 0; n < OUTNODE; n++) {
            double sum = 0;
            for (size_t k = 0; k < HIDENODE_3; k++) {
                sum += ::hideLayer_3[k]->value * ::hideLayer_3[k]->weight[n];
            }
            sum -= ::outLayer[n]->bias;

            ::outLayer[n]->value = sum;

            idx.out.push_back(::outLayer[n]->value);

           
           for (auto& tmp : idx.in) {
                cout << tmp << " ";
            }
            

            for (auto& tmp : idx.out) {
                cout << tmp << " ";
            }
            cout << endl;
        }
    }
    return 0;
}
