'use strict'

const express = require('express')

const app = express()
const path = require('path')
const fs = require('fs')

const { exec } = require('child_process')

const multer  = require('multer')
const upload = multer({ dest: 'uploads/' })

app.set('view engine', 'ejs')

app.use(express.static(path.resolve(__dirname, 'static')))

app.get('/', (req, res, next) => {

    fs.readdir(path.resolve(__dirname, '../Models'), (err, files) => {

        if (err)
        return next(err)
        
        res.render(path.resolve(__dirname, 'index.ejs'), {
            models: files
        })
    })
})

app.post('/predict', upload.single('file'),  (req, res, next) => {

    const modelName = req.body.model
    const filePath = path.resolve(__dirname, req.file.path)

    exec(`python predict.py ${filePath} ${modelName}`, (error, stdout, stderr) => {

        fs.unlink(filePath, (err) => {
            if (err)
            console.error(err.message)
        })
        
        if (error){
            console.error(stderr)
            return res.status(500).json({ error: error.message })
        }

        try {
            const result = parseInt(stdout)
            return res.json({ result })
        } 
        
        catch (e){
            return res.status(500).json({ error: e.message })
        }
    })
})

app.listen(80, () => {
    console.log("application is listening.")
})