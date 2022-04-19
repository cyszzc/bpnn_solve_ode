#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>
#include <random>
using namespace std;

#define INNODE 1
#define HIDENODE 4
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

Node* inputLayer[INNODE], * hideLayer[HIDENODE], * outLayer[OUTNODE];

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
        for (size_t j = 0; j < OUTNODE; j++) {
            ::hideLayer[i]->weight.push_back(distribution(rd));
            ::hideLayer[i]->weight_delta.push_back(0.f);
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

            for (size_t j = 0; j < OUTNODE; j++) {
                double sum = 0;
                for (size_t i = 0; i < HIDENODE; i++) {
                    sum += ::hideLayer[i]->value * ::hideLayer[i]->weight[j];
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


            for (size_t i = 0; i < OUTNODE; i++) {
                double bias_delta = -(idx.out[i] - ::outLayer[i]->value);
                ::outLayer[i]->bias_delta += bias_delta;
            }

            for (size_t i = 0; i < HIDENODE; i++) {
                for (size_t j = 0; j < OUTNODE; j++) {
                    double weight_delta = (idx.out[j] - ::outLayer[j]->value) * ::hideLayer[i]->value;
                    ::hideLayer[i]->weight_delta[j] += weight_delta;
                }
            }

            for (size_t i = 0; i < HIDENODE; i++) {
                double sum = 0;
                for (size_t j = 0; j < OUTNODE; j++) {
                    sum += -(idx.out[j] - ::outLayer[j]->value) * ::hideLayer[i]->weight[j];
                }
                ::hideLayer[i]->bias_delta += sum * ::hideLayer[i]->value * (1.0 - ::hideLayer[i]->value);
            }

            for (size_t i = 0; i < INNODE; i++) {
                for (size_t j = 0; j < HIDENODE; j++) {
                    double sum = 0.f;
                    for (size_t k = 0; k < OUTNODE; k++) {
                        sum += (idx.out[k] - ::outLayer[k]->value) * ::hideLayer[j]->weight[k];
                    }
                    ::inputLayer[i]->weight_delta[j] +=
                        sum *
                        ::hideLayer[j]->value * (1.0 - ::hideLayer[j]->value) *
                        ::inputLayer[i]->value;
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
            for (size_t j = 0; j < OUTNODE; j++) {
                ::hideLayer[i]->weight[j] +=
                    rate * ::hideLayer[i]->weight_delta[j] / train_data_size;
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

        for (size_t i = 0; i < INNODE; i++) {
            ::inputLayer[i]->value = idx.in[i];
        }

        for (size_t j = 0; j < HIDENODE; j++) {
            double sum = 0;
            for (size_t i = 0; i < INNODE; i++) {
                sum += ::inputLayer[i]->value * inputLayer[i]->weight[j];
            }
            sum -= ::hideLayer[j]->bias;

            ::hideLayer[j]->value = utils::sigmoid(sum);
        }

        for (size_t j = 0; j < OUTNODE; j++) {
            double sum = 0;
            for (size_t i = 0; i < HIDENODE; i++) {
                sum += ::hideLayer[i]->value * ::hideLayer[i]->weight[j];
            }
            sum -= ::outLayer[j]->bias;

            ::outLayer[j]->value = sum;

            idx.out.push_back(::outLayer[j]->value);
            
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
