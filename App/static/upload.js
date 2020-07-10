(function (){

    const resetBoxes = () => {
        $("#error-box").addClass('d-none')
        $("#success-box").addClass('d-none')
    }

    const uploadError = (error) => {
        resetBoxes()

        const box = $("#error-box")

        box.text("Error: " + error)
        box.removeClass('d-none')
    }

    const uploadSuccess = (type) => {
        resetBoxes()

        const box = $("#success-box")

        switch (type){
            case 0:
                box.text('This is an orange !')
                break
            case 1:
                box.text('This is a banana !')
                break
            case 2:
                box.text('This is a kiwi !')
                break
        }

        box.removeClass('d-none')
    }

    $(document).ready(function (){

        const form = document.getElementById('upload-form')

        form.onsubmit =  async function (ev){

            resetBoxes()
            
            ev.preventDefault()
            
            const response = await fetch('/predict', {
                method: 'POST',
                body: new FormData(form)
            })

            const json = await response.json()

            if (response.status !== 200){
                return uploadError(json['error'])
            }

            return uploadSuccess(json['result'])
            
        }
    })
}())