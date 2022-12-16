# MiniProject5 - Improving Employee Retention by Predicting Employee Attrition Using Machine Learning
Rheza Paleva Uyanto - Rakamin Academy Data Science Batch 24

## Latar Belakang
“Sumber daya manusia (SDM) adalah aset utama yang perlu dikelola dengan baik oleh perusahaan agar tujuan bisnis dapat tercapai dengan efektif dan efisien. Pada kesempatan kali ini, kita akan menghadapi sebuah permasalahan tentang sumber daya manusia yang ada di perusahaan. Fokus kita adalah untuk mengetahui bagaimana cara menjaga karyawan agar tetap bertahan di perusahaan yang ada saat ini yang dapat mengakibatkan bengkaknya biaya untuk rekrutmen karyawan serta pelatihan untuk mereka yang baru masuk. Dengan mengetahui faktor utama yang menyebabkan karyawan tidak merasa, perusahaan dapat segera menanggulanginya dengan membuat program-program yang relevan dengan permasalahan karyawan.“

## Pertanyaan Riset
- Faktor utama yang menyebabkan tingkat resign menjadi tinggi ?
- Apa alasan tertinggi karyawan resign ?

## Tujuan
Untuk mengetahui faktor apa saja yang menyebabkan karyawan resign.

## Data Preprocessing :
- Dataset : Improving Employee Retention by Predicting Employee Attrition Using Machine Learning.xlsx - hr_data.csv
- Terdiri dari 25 kolom dan 287 baris.

### Handle Null Value :
- SkorKepuasanPegawai	: 5 NaN diisi median
- JumlahKeikutsertaanProjek : 3 NaN diisi median
- JumlahKeterlambatanSebulanTerakhir : 1 NaN, diisi median
- JumlahKetidakhadiran 	: 6 NaN, diisi median
- AlasanResign 	: 66 NaN, diisi ‘masih bekerja’

### Fitur yang berbeda:
- StatusPernikahan , tidak ada null value : 3 terisi ‘-’, diganti ke ‘Lainnya’ , karena bermakna sama.

### Fitur yang di drop:
- PernahBekerja, tidak ada null value tetapi ada 1 baris yang berisi ‘yes’ sedangkan lainnya 1, 1 dan yes bermakna sama sehingga hanya terdiri dari 1 nilai, kolom didrop
- IkutProgramLOP terdapat 258 null dari 287 value, lebih dari 50%, sehingga kolom didrop.
- Username, karena kardinalitas tinggi
- NomorHP, karena kardinalitas tinggi
- Email, karena kardinalitas tinggi

### Nilai Duplikat : 
- tidak ditemukan

## Data Visualisation
### Annual Report on Employee Number Changes
Hal yang dilakukan :
- Untuk mendapat grafik pertumbuhan kartawan setiap tahun, diperlukan proses agregasi data
- Melakukan ekstrasi data tanggal_hiring untuk mendapatkan tahun_hiring
- Melakukan aggregasi dengan groupby sehingga didapatkan jumlah karyawan yang dihiring pertahun.
- Melakukan hal yang sama untuk mendapatkan tahun_resign
- Melakukan outer join, untuk menggabungkan tabel hiring dan resign

<img width="650" alt="image" src="https://user-images.githubusercontent.com/114345988/208027801-ebd8690a-c4f6-4533-81ed-987e38b9bf21.png">

Penjelasan:
- Berdasarkan waterfall plot, pada 10 tahun pertama (2006-2016), jumlah karyawan yang direkrut terus bertambah dibandingkan dengan jumlah karyawan yang resign. 
- Pertumbuhan jumlah karyawan yang paling tinggi sebanyak 76 orang terjadi pada tahun 2011. Sampai pada tahun 2016, pertumbuhan jumlah karyawan berhenti dengan bertambahnya 6 orang. Pada akhir tahun 2016,jumlah karyawan sebanyak 248 orang
- Setelah tahun 2016, karyawan yang resign lebih banyak. Karyawan yang resign paling banyak terjadi pada tahun 2018, yakni berkurang 25 orang. 
- Sampai tahun 2020, total karyawan yang resign terus bertambah, sehingga akhir tahun 2020, jumlah karyawan sebanyak 198 orang Dari tahun 2016 sampai akhir tahun 2020, jumlah karyawan berkurang sebanyak 20,16% atau sebanyak 50 orang.

### Resign Reason Analysis for Employee Attrition Management Strategy
Hal yang dilakukan :
- Melakukan aggregasi jumlah karyawan yang resign dengan groupby per jenis pekerjaan.
- Melakukan aggregasi jumlah karyawan yang belum resign dengan groupby per jenis pekerjaan.
- Melakukan outer join, untuk menggabungkan tabel hiring dan resign.
- Mengisi null-value pada kolom yang sesuai,
- Membuat kolom baru untuk menjumlahkan jumlah karyawan yang resign dan belum resign
- Menghitung persentase %resign, dari masing-masing bidang pekerjaan

<img width="450" alt="image" src="https://user-images.githubusercontent.com/114345988/208028268-153b73f6-f452-4563-b3a8-cfca55c71ed5.png">

Penjelasan :
- Tingkat resign yang tertinggi berada pada bidang pekerjaan Data Analyst yakni 50 %, diikuti oleh Software Engineer (Front End) 38,89%, dan Product Design (UI dan UX) 37,50%
- Jumlah karyawan sebagai Data Analyst yang resign sebanyak 8 orang. 

<img width="450" alt="image" src="https://user-images.githubusercontent.com/114345988/208028467-fa6184af-33c0-4457-9fc7-15b0eacb7c34.png">

Penjelasan :
- Dari 8 orang Data Analyst yang resign, kita perlu mengetahui latar belakang mereka berdasarkan Jenjang Karir, Alasan Resign dan Performance selama bekerja.
- Berdasarkan jenjang karir, ke-delapan data analyst yang resign berasal dari jenjang freshgraduate.
- Mereka yang resign mayoritas beralasan toxic culture dan alasan lainnya seperti internal confict. Kedua alasan tersebut merupakan alasan yang berasal dari kondisi bekerja yang tidak nyaman. 
- Lima dari Delapan karyawan memiliki performance pekerjaan dari bagus sampai sangat bagus. Hal ini cukup disayangkan, karena dari sisi pekerjaan, mereka dapat diandalkan, tetapi memilih untuk resign.
- Untuk Manajemen, perlu melakukan root cause analysis penyebab adanya toxic culture di perusahaan, apakah dari sisi perusahaannya atau sisi perilaku karyawannya. 

## Build an Automated Resignation Behavior Prediction using Machine Learning
### Dilakukan Feature Engineering lanjutan:
- usia_hiring : mengurangkan tahun lahir dan tahun hiring
- lama_kerja : mengurangkan tahun saat ini atau tahun saat resign dengan tahun hiring
- JumlahKeikutsertaanProjek : diberi 0 untuk ikut serta, 1 untuk tidak ikut serta
- KategoriPekerjaan : mengelompokkan pekerjaan dalam bidang software, data, dan product
### Handling Outlier
- Sebelum handlier outlier : 278 value, sesudah handling outlier : 273 value
### Dropping feature sebelum feature encoding:
- TanggalLahir, TanggalHiring, TanggalPenilaianKaryawan, TanggalResign, tahun_hiring, tahun_resign, tahun_lahir, dan JumlahKeikutsertaanProjek
### Feature Encoding
- Frequency Encoding : HiringPlatform
- Label Encoding : Performance Pegawai, Tingkat Pendidikan, KategoriPekerjaan dan Jenjang karir
- One Hot Encoding : Status Pernikahan, Jenis Kelamin, Status Kepegawaian, dan Asal Daerah.
### Dropping feature sesudah label encoding:
- EnterpriseID, StatusPernikahan, JenisKelamin, StatusKepegawaian, Pekerjaan, KategoriPekerjaan, AsalDaerah, HiringPlatform, dan AlasanResign
### Null Value :
- tidak ada
### Duplicated Value:
- tidak ada

### Split Data Train dan data Test
- Perbandingan 8:2
- y_train memiliki perbandingan kelas yang tidak seimbang
- Selanjutnya dilakukan handling  imbalanced class menggunakan 5 teknik: RandomUndersampler, TomekLinks, EditedNearestNeighbours, SMOTEENN, dan SMOTETomek
- Untuk Modelling Machine Learning menggunakan model XGBoost Classifier
- Untuk menggabungkan Handling Imbalanced Class dan Model Machine Learning menggunakan Pipeline.
- Untuk mengevaluasi Handling Imbalanced Class dan Model Machine Learning menggunakan RepeatedStratifiedKFold, dengan nilai accuracy, precision dan recall score.
- Score adalah sebagai berikut:
<img width="593" alt="image" src="https://user-images.githubusercontent.com/114345988/208029874-9edd4153-427b-4b96-a1bd-00d56530bde8.png">

- Nilai ROC
<img width="450" alt="image" src="https://user-images.githubusercontent.com/114345988/208029401-0525b5d5-5648-4657-b440-2af5b3778b39.png">
nilai ROC dengan handling imbalance class Tomek Link dan XG Boost lebih baik

